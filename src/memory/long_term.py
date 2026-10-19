# ============================================================
# 长期记忆（语义召回）
# ------------------------------------------------------------
# 与文件知识库是"同一套 RAG 技术"，但存的是"历史问答"而非文档。
# 好处：当用户问"上次那个方案再讲一遍"时，能靠语义把旧对话捞出来，
# 而不用把所有历史都塞进上下文。这就是"长期记忆"的工程实现。
#
# 【本版修复】旧版长期记忆只活在内存里——服务一重启，用户之前
# 告诉过助手的所有信息全部丢失，用户会感觉"记忆功能根本没做"。
# 现在改为 JSONL 落盘（data/.memory/long_term.jsonl），启动自动加载，
# 并且重新加载时的向量化会命中嵌入磁盘缓存，不额外花 token。
# ============================================================
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings


def _looks_garbled(text: str) -> bool:
    """判断一段文本是不是编码损坏的乱码。

    真实踩过的坑：用某些客户端（PowerShell / 早期版本 curl）以非 UTF-8
    编码发送中文，服务端收到的是一串 "?"。这类内容一旦被存进长期记忆，
    就会污染召回——问任何问题都可能捞到这些乱码，进而让模型答非所问。
    所以在**入口**就挡住，比事后清理便宜得多。

    判定规则：去掉空白后，问号等 ASCII 标点占比过高即视为乱码。
    """
    s = (text or "").strip()
    if not s:
        return False
    # 连续的 ??/??? 是编码丢失的典型特征
    if re.search(r"[?？]{2,}", s):
        return True
    # ASCII 标点（? ! . , 等）占比 > 40% 且中文字符很少
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    punct = sum(1 for ch in s if ch in "?!.?,，。？！、")
    if cjk == 0 and punct / max(1, len(s)) > 0.4:
        return True
    return False


class LongTermMemory:
    """把每轮问答向量化存储，按语义相似度召回。"""

    def __init__(
        self,
        embeddings: Embeddings,
        top_k: int = 2,
        persist_path: str | Path | None = None,
        max_items: int = 500,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_items = max_items
        self._texts: list[str] = []      # 记忆原文
        self._matrix: np.ndarray | None = None
        if self.persist_path is not None:
            self._load()

    # ------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------
    def add(self, question: str, answer: str) -> None:
        """把一轮问答拼成一条记忆存起来（同时落盘）。"""
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return
        # 乱码不入库：防止编码异常的请求污染长期记忆（见 _looks_garbled 说明）
        if _looks_garbled(question):
            print(f"[memory] 跳过一条乱码记忆：{question[:20]!r}")
            return
        memory_text = f"Q: {question}\nA: {answer}"

        # 同一句话不重复记（避免反复问同一问题时记忆膨胀）
        if memory_text in self._texts:
            return

        self._texts.append(memory_text)
        vecs = np.array(self.embeddings.embed_documents([memory_text]), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self._matrix = vecs if (self._matrix is None or len(self._matrix) == 0) \
            else np.vstack([self._matrix, vecs])

        self._append_disk(question, answer)

    def recall(self, query: str) -> list[str]:
        """按语义召回最相关的 N 条历史记忆。"""
        if self._matrix is None or not self._texts:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        # 做矩阵相似度：embed_query 已归一化，但为稳妥这里统一按点积算
        scores = self._matrix @ q
        n = min(self.top_k, len(self._texts))
        idx = np.argsort(scores)[::-1][:n]
        return [self._texts[i] for i in idx if 0 <= i < len(self._texts)]

    def clear(self) -> None:
        self._texts.clear()
        self._matrix = None
        if self.persist_path is not None and self.persist_path.exists():
            self.persist_path.unlink()

    def __len__(self) -> int:
        return len(self._texts)

    # ------------------------------------------------------------
    # 落盘 / 加载
    # ------------------------------------------------------------
    def _append_disk(self, question: str, answer: str) -> None:
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"q": question, "a": answer, "ts": int(time.time())},
                    ensure_ascii=False,
                ) + "\n")
            if len(self._texts) > self.max_items:
                self._rewrite_disk()
        except Exception as e:
            print(f"[memory] 写入长期记忆失败（不影响本次运行）：{e}")

    def _load(self) -> None:
        """启动时加载历史记忆；向量化会命中嵌入缓存，不额外消耗 token。"""
        path = self.persist_path
        if path is None or not path.exists():
            return
        try:
            items: list[tuple[str, str]] = []
            skipped = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                q, a = obj.get("q", ""), obj.get("a", "")
                if not q or not a:
                    skipped += 1
                elif _looks_garbled(q):     # 跳过历史遗留的乱码脏数据
                    skipped += 1
                else:
                    items.append((q, a))
            if not items:
                return
            items = items[-self.max_items:]
            self._texts = [f"Q: {q}\nA: {a}" for q, a in items]
            vecs = np.array(self.embeddings.embed_documents(self._texts), dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            self._matrix = vecs
            tail = f"（跳过 {skipped} 条脏数据）" if skipped else ""
            print(f"[memory] 已加载 {len(self._texts)} 条长期记忆{tail}")
        except Exception as e:
            print(f"[memory] 加载长期记忆失败（将从头开始）：{e}")
            self._texts, self._matrix = [], None

    def _rewrite_disk(self) -> None:
        """超过上限时重写文件，只保留最近的 max_items 条。"""
        if self.persist_path is None:
            return
        keep = self._texts[-self.max_items:]
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                for text in keep:
                    q, _, a = text.partition("\nA: ")
                    f.write(json.dumps(
                        {"q": q[3:], "a": a, "ts": int(time.time())},
                        ensure_ascii=False,
                    ) + "\n")
            self._texts = keep
        except Exception as e:
            print(f"[memory] 重写长期记忆失败：{e}")
