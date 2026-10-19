# ============================================================
# 嵌入向量磁盘缓存（省 token 的关键）
# ------------------------------------------------------------
# 问题：一次问答要花 15 次请求、2 万多 token，其中绝大部分不是
# "回答"花的，而是**建索引**花的——每个文档块都要调一次嵌入接口。
# 更要命的是：每次重启服务都会把同样的文档重新嵌入一遍，钱白烧。
#
# 做法：把"文档块文本 -> 向量"按**内容 hash** 缓存到磁盘。
#   - 内容没变 => 直接读缓存，0 次嵌入请求；
#   - 只新增了一个文件 => 只有这个文件的新块会 miss，其他全命中（增量索引）；
#   - 换了嵌入模型 => namespace 变化，自动失效重算，不会串味。
#
# 缓存键 = sha1(namespace + 文本)，namespace = sha1(模型名 + 端点)。
# 文件布局：cache/emb_<ns>.npy（float32 矩阵） + cache/emb_<ns>.json（键顺序）
# ============================================================
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings


class EmbeddingCache:
    """嵌入向量的磁盘缓存：key(文本hash) -> 向量。"""

    def __init__(self, cache_dir: str | Path, namespace: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._npy = self.dir / f"emb_{namespace}.npy"
        self._json = self.dir / f"emb_{namespace}.json"
        self._keys: list[str] = []
        self._rows: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._dirty = False
        self.hits = 0
        self.misses = 0
        self._load()

    # ------------------------------------------------------------
    @staticmethod
    def make_namespace(model: str, base_url: str) -> str:
        """按"模型 + 端点"生成命名空间：换模型时缓存自动隔离。"""
        raw = f"{model}|{base_url}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    @staticmethod
    def make_key(text: str) -> str:
        """缓存键：只看文本内容，所以同一段文字在不同文件里也能复用。"""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------
    def _load(self) -> None:
        """启动时一次性把缓存读进内存。"""
        if not (self._npy.exists() and self._json.exists()):
            return
        try:
            meta = json.loads(self._json.read_text(encoding="utf-8"))
            keys = meta.get("keys", [])
            matrix = np.load(self._npy)
            if not keys or matrix.ndim != 2 or matrix.shape[0] != len(keys):
                return   # 元数据与矩阵不一致，视为无缓存
            self._keys = keys
            self._rows = {k: i for i, k in enumerate(keys)}
            self._matrix = matrix.astype(np.float32)
        except Exception as e:
            print(f"[cache] 读取缓存失败（将重新计算）：{e}")

    def flush(self) -> None:
        """把新增的向量落盘。"""
        if not self._dirty or self._matrix is None:
            return
        try:
            np.save(self._npy, self._matrix)
            self._json.write_text(
                json.dumps({"keys": self._keys, "dim": int(self._matrix.shape[1])},
                           ensure_ascii=False),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as e:
            print(f"[cache] 写缓存失败（不影响本次运行）：{e}")

    # ------------------------------------------------------------
    def get(self, key: str) -> np.ndarray | None:
        if self._matrix is None:
            return None
        row = self._rows.get(key)
        if row is None:
            return None
        return self._matrix[row]

    def put(self, key: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if self._matrix is None or len(self._matrix) == 0:
            self._matrix = vec
        else:
            if vec.shape[1] != self._matrix.shape[1]:
                return   # 维度不一致（换过模型），跳过
            self._matrix = np.vstack([self._matrix, vec])
        self._rows[key] = len(self._keys)
        self._keys.append(key)
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._keys)

    def stats(self) -> str:
        return f"缓存命中 {self.hits} 条 / 新算 {self.misses} 条（库内共 {self.size} 条）"


class CachedEmbeddings(Embeddings):
    """给任何 Embeddings 套一层磁盘缓存。

    对外仍是标准 Embeddings 接口，所以上层（向量库、长期记忆）完全无感。
    查询向量（embed_query）不缓存——它每次都不同，缓存没有意义。
    """

    def __init__(self, inner: Embeddings, cache: EmbeddingCache):
        self.inner = inner
        self.cache = cache

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self.cache.make_key(t) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []

        for i, k in enumerate(keys):
            v = self.cache.get(k)
            if v is None:
                missing.append(i)
            else:
                out[i] = v
                self.cache.hits += 1

        if missing:
            self.cache.misses += len(missing)
            fresh = self.inner.embed_documents([texts[i] for i in missing])
            for j, i in enumerate(missing):
                v = np.asarray(fresh[j], dtype=np.float32)
                out[i] = v
                self.cache.put(keys[i], v)
            self.cache.flush()

        return [v.tolist() for v in out]  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        """查询向量直接透传，不进缓存（每次查询都不同，缓存没有意义）。

        注意：Embeddings 是抽象基类，embed_query 必须实现，
        否则实例化时就会抛 TypeError（这条曾被单元测试抓到）。
        """
        return self.inner.embed_query(text)
