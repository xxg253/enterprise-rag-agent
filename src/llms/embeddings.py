# ============================================================
# 嵌入模型层（Embeddings）
# ------------------------------------------------------------
# 嵌入（Embedding）把文本变成高维向量，是 RAG "语义检索"的前提：
# 用户问题和文档片段都被映射到同一向量空间，靠余弦相似度找"意思相近"的内容。
# 同样提供 Mock（确定性伪向量）与真实 OpenAI 兼容两种实现。
# ============================================================
from __future__ import annotations

import hashlib
import math

import numpy as np
from langchain_core.embeddings import Embeddings

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None

from src.config import EmbeddingConfig


class MockEmbeddings(Embeddings):
    """离线 Mock 嵌入。

    重要说明：它用字符哈希生成一个确定性的伪向量，**不具备语义相似性**
    （"苹果"和"水果"的向量不会真的更接近）。它的作用只是让 RAG 流水线
    在无 Key 时也能跑通、让你看到"检索→生成"的完整数据流动。
    真实场景中请务必使用语义嵌入模型（如 text-embedding-v3）。
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        """把文本映射成 dim 维向量：对每个字符哈希后做一个归一化的"词袋"。"""
        vec = [0.0] * self.dim
        for ch in text:
            h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        # L2 归一化，方便后续直接做余弦相似度（点积即余弦）
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class BatchedEmbeddings(Embeddings):
    """分批嵌入包装器：把一个大批次拆成多个小批次依次提交。

    为什么需要它：很多嵌入服务对**单次请求的文本条数**有上限
    （例如阿里云 text-embedding-v3 限制 batch size ≤ 10）。
    如果建索引时把几十个文档块一次性提交，接口会直接返回 400：
      "batch size is invalid, it should not be larger than 10"
    从而让整个知识库初始化失败。

    这个包装器把任意底层嵌入模型包一层，对外仍是标准 Embeddings 接口，
    内部自动按 batch_size 切片，既解决限制又不影响上层代码。
    """

    def __init__(self, inner: Embeddings, batch_size: int = 10):
        self.inner = inner
        self.batch_size = max(1, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批调用底层模型，最后拼回完整结果（顺序保持不变）。"""
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend(self.inner.embed_documents(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        """查询向量只有一条，直接透传。"""
        return self.inner.embed_query(text)


def get_embeddings(cfg: EmbeddingConfig) -> Embeddings:
    """嵌入模型工厂，逻辑与 get_chat_model 一致（依赖注入）。

    重要：必须传 `tiktoken_enabled=False`。
    原因：LangChain 的 OpenAIEmbeddings 默认会先用 tiktoken 对文本分词，
    而 tiktoken 不认识自定义模型名（如 text-embedding-v3），会尝试
    **联网下载分词表**，在离线/被墙环境下抛 SSLError，导致服务初始化失败。
    关掉它后，文本会原样发给嵌入接口，由服务端自己分词，既正确又不联网。

    另外统一用 BatchedEmbeddings 包一层，解决"单次最多 10 条"的接口限制。
    """
    if cfg.provider == "openai" and OpenAIEmbeddings is not None:
        kwargs = dict(
            model=cfg.model,
            base_url=cfg.base_url or None,
            api_key=cfg.api_key or "EMPTY",
            # 关键：禁用本地 tiktoken 分词，避免联网下载分词表
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            # 关键：设置超时，避免网络问题导致建索引无限挂起
            timeout=60,
            max_retries=1,
        )
        try:
            inner = OpenAIEmbeddings(**kwargs)
        except TypeError:
            # 兼容某些版本的参数名差异
            inner = OpenAIEmbeddings(
                model=cfg.model,
                base_url=cfg.base_url or None,
                api_key=cfg.api_key or "EMPTY",
            )
        # 包一层分批器：默认每批 10 条（满足主流嵌入服务的限制）
        return BatchedEmbeddings(inner, batch_size=10)
    return MockEmbeddings()