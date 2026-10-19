# ============================================================
# 检索器（Retriever）
# ------------------------------------------------------------
# 对上层的统一检索入口：把"向量库 + 嵌入 + 检索策略"封装成一个对象，
# Agent 只需要调用 .retrieve(query)，不用关心底层是 Milvus 还是 numpy。
#
# 【本版重点修复】"知识库里有答案却只召回一部分"
#   旧版直接取向量相似度前 top_k 个，有两个硬伤：
#     1) 大文件霸榜 —— BS.pdf 有 133 块，Top-K 常常被它一个文件占满，
#        简历（只有 5 块）根本挤不进去，所以"两段实习"召不全；
#     2) 信息被切在块边界 —— 命中的块只有半句话，另外半句在隔壁块里，
#        模型看到残缺内容就答"上下文未提供"。
#   对策（两招，都是工业界常规做法）：
#     A. 来源均衡：先多召回候选，再做"单一来源配额"，保证 Top-K 覆盖多个文件；
#     B. 邻块扩展：命中某块后，自动把同源的相邻块一并带上，
#        让被切断的句子/段落重新拼完整。
# ============================================================
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from src.config import RetrievalConfig
from src.rag.loader import Document, load_and_split
from src.rag.vectorstore import VectorStore, build_vectorstore


@dataclass
class RetrievedChunk:
    """检索结果：文档块 + 相似度分数。"""
    content: str
    source: str
    score: float
    chunk_id: int = -1
    expanded: bool = False   # 是否由"邻块扩展"补进来的（不是直接命中）


class Retriever:
    """RAG 检索器：索引构建 + 相似度召回 + 来源均衡 + 邻块扩展。"""

    def __init__(self, cfg: RetrievalConfig, embeddings: Embeddings, root=None):
        self.cfg = cfg
        self.embeddings = embeddings
        # 向量库工厂需要完整配置（Milvus 的 uri / 集合名 / 索引类型）
        self.store: VectorStore = build_vectorstore(
            cfg.vectorstore, embeddings, retrieval_cfg=cfg, root=root
        )
        # (来源文件, chunk_id) -> Document，供邻块扩展按坐标快速取块
        self._index: dict[tuple[str, int], Document] = {}
        self._ready = False

    @property
    def backend(self) -> str:
        """实际生效的向量库后端名（milvus / numpy），用于前端如实展示。"""
        return getattr(self.store, "backend", "numpy")

    @property
    def collection(self) -> str:
        """向量库集合名（numpy 版没有集合概念，返回空串）。"""
        return getattr(self.store, "collection", "")

    @property
    def vectors(self) -> int:
        """库里实际有多少条向量（Milvus 侧的真实统计）。"""
        try:
            return len(self.store)
        except Exception:
            return 0

    @property
    def index_type(self) -> str:
        """向量索引类型（HNSW / IVF_FLAT），仅 Milvus 后端有。"""
        return getattr(self.store, "index_type", "")

    @property
    def metric(self) -> str:
        """相似度度量（IP 内积 / L2），仅 Milvus 后端有。"""
        return getattr(self.store, "metric", "")

    # ------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------
    def index_directory(self, data_dir: str) -> int:
        """把某个目录下的文档全部建索引。返回索引的块数。"""
        documents = load_and_split(
            data_dir,
            self.cfg.chunk_size,
            self.cfg.chunk_overlap,
            getattr(self.cfg, "min_chunk_file", 2000),
        )
        return self.index_documents(documents)

    def index_documents(self, documents: list[Document]) -> int:
        """直接索引一组给定文档（测试/生产导入知识时使用）。"""
        if documents:
            self.store.add(documents)
            for d in documents:
                self._index[(d.source, d.chunk_id)] = d
        self._ready = True
        return len(documents)

    @property
    def doc_count(self) -> int:
        return len(self._index)

    def sources(self) -> list[str]:
        """知识库里有哪些来源文件（用于日志/前端展示）。"""
        return sorted({s for s, _ in self._index})

    # ------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """核心检索：返回最相关的若干文档块（已做来源均衡 + 邻块扩展）。"""
        k = max(1, top_k or self.cfg.top_k)
        if self.store is None or self.doc_count == 0:
            return []

        # 1) 先多召回候选（top_k 的 4 倍），为后续重排/均衡留出空间
        candidates = self.store.search(query, k * 4)
        min_score = getattr(self.cfg, "min_score", 0.0) or 0.0
        filtered = [(d, s) for d, s in candidates if s > min_score]
        if not filtered:                       # 全被过滤掉就退回原始候选
            filtered = candidates

        # 2) 来源均衡：单一文件最多占 max_per_source 个名额，
        #    避免 133 块的大文件把 5 块的小文件挤出 Top-K
        picked = self._diversify(filtered, k)

        # 3) 邻块扩展：把命中块的左右邻居补齐，修复"被切在边界"的信息
        return self._expand(picked)

    # ------------------------------------------------------------
    def _diversify(self, hits: list[tuple[Document, float]], k: int) -> list[tuple[Document, float]]:
        """按来源配额挑选 Top-K，保证结果覆盖多个文件。"""
        max_per = max(1, int(getattr(self.cfg, "max_per_source", 2) or 2))
        per: Counter[str] = Counter()
        picked: list[tuple[Document, float]] = []

        for doc, score in hits:
            if per[doc.source] >= max_per:
                continue
            picked.append((doc, score))
            per[doc.source] += 1
            if len(picked) >= k:
                return picked

        # 配额还没凑够 K 条：放宽限制，按分数补齐（保证召回数量）
        seen = {(d.source, d.chunk_id) for d, _ in picked}
        for doc, score in hits:
            if len(picked) >= k:
                break
            if (doc.source, doc.chunk_id) in seen:
                continue
            picked.append((doc, score))
            seen.add((doc.source, doc.chunk_id))
        return picked

    def _expand(self, picked: list[tuple[Document, float]]) -> list[RetrievedChunk]:
        """邻块扩展：命中块 ±radius 的同源块一起返回。

        输出按「来源文件分组、chunk_id 升序」排列，
        这样拼进提示词时上下文顺序与原文一致，模型读起来不会错位。
        """
        radius = int(getattr(self.cfg, "neighbor_expand", 0) or 0)
        keep: dict[tuple[str, int], float] = {}

        for doc, score in picked:
            for off in range(-radius, radius + 1):
                key = (doc.source, doc.chunk_id + off)
                if key not in self._index:
                    continue
                # 直接命中的用原分；邻居略微降权，保证排序仍以命中块为主
                s = score if off == 0 else score * 0.98
                if s > keep.get(key, -1.0):
                    keep[key] = s

        by_source: dict[str, list[int]] = {}
        for (src, cid) in keep:
            by_source.setdefault(src, []).append(cid)

        results: list[RetrievedChunk] = []
        # 来源之间按各自最高分排序，文件内部按 chunk_id 升序（还原原文顺序）
        ordered_sources = sorted(
            by_source.keys(),
            key=lambda s: -max(keep[(s, c)] for c in by_source[s]),
        )
        for src in ordered_sources:
            hit_ids = {d.chunk_id for d, _ in picked if d.source == src}
            for cid in sorted(by_source[src]):
                doc = self._index.get((src, cid))
                if doc is None:
                    continue
                results.append(
                    RetrievedChunk(
                        content=doc.content,
                        source=src,
                        score=round(keep[(src, cid)], 4),
                        chunk_id=cid,
                        expanded=cid not in hit_ids,
                    )
                )
        return results
