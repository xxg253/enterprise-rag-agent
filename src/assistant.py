# ============================================================
# 顶层助手（编排层）
# ------------------------------------------------------------
# 把三大件组装成一个"会话式助手"：
#   RAG 知识库  +  短期/长期记忆  +  LangGraph 多智能体图
# 对外只暴露一个 ask(question) 方法，内部完成：
#   读记忆 -> 跑图 -> 更新记忆 -> 返回答案
# 这就是"企业级 Agent"对外应有的干净接口。
#
# 【本版增强】
#   1) 注入嵌入磁盘缓存（CachedEmbeddings），重建索引时命中的块
#      不再重复调用嵌入接口 —— 这是省 token 的最大头；
#   2) 长期记忆落盘（JSONL），重启服务后依然记得用户说过什么；
#   3) 建索引时把 min_chunk_file 传下去，小文件整篇不切。
# ============================================================
from __future__ import annotations

from pathlib import Path

from src.agents.graph import build_graph
from src.config import AppConfig
from src.llms.embeddings import get_embeddings
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.rag.cache import CachedEmbeddings, EmbeddingCache
from src.rag.retriever import Retriever


class EnterpriseAssistant:
    """企业知识助手：RAG + 记忆 + 多智能体推理。"""

    def __init__(self, cfg: AppConfig, root: str | Path | None = None):
        self.cfg = cfg
        # 项目根目录：用于定位缓存目录等相对路径
        self.root = Path(root) if root else Path.cwd()

        # 1) 嵌入模型（被 RAG 与长期记忆共用），可选地套一层磁盘缓存
        self.embedding_cache: EmbeddingCache | None = None
        self.embeddings = self._build_embeddings()

        # 2) RAG 检索器（传 root 让向量库把本地数据文件挂到项目目录下）
        self.retriever = Retriever(cfg.retrieval, self.embeddings, root=self.root)

        # 3) 记忆：短期在内存，长期可落盘
        self.short_term = ShortTermMemory(cfg.memory.short_term_window)
        self.long_term: LongTermMemory | None = None

        # 4) 多智能体图（延迟到索引建好后再编译，见 build）
        self.graph = None
        self._data_dir: Path | None = None

    # ------------------------------------------------------------
    def _build_embeddings(self):
        """按配置构建嵌入模型；开启缓存时用 CachedEmbeddings 包一层。

        缓存命名空间 = f(模型名, 端点)，换模型会自动隔离，不会串味。
        """
        base = get_embeddings(self.cfg.embeddings)
        if not getattr(self.cfg.embeddings, "cache", False):
            return base

        cache_dir = Path(self.cfg.embeddings.cache_dir or "cache")
        if not cache_dir.is_absolute():
            cache_dir = self.root / cache_dir
        ns = EmbeddingCache.make_namespace(
            self.cfg.embeddings.model, self.cfg.embeddings.base_url
        )
        self.embedding_cache = EmbeddingCache(cache_dir, ns)
        if self.embedding_cache.size:
            print(f"[cache] 已加载嵌入缓存 {self.embedding_cache.size} 条（{cache_dir}）")
        return CachedEmbeddings(base, self.embedding_cache)

    def _build_long_term(self, data_dir: str) -> LongTermMemory:
        """构建长期记忆；persist=True 时落盘到 <data_dir>/.memory/long_term.jsonl。"""
        persist_path = None
        if getattr(self.cfg.memory, "persist", False):
            persist_path = Path(data_dir) / (self.cfg.memory.memory_dir or ".memory") / "long_term.jsonl"
        return LongTermMemory(
            self.embeddings,
            self.cfg.memory.long_term_top_k,
            persist_path=persist_path,
        )

    # ------------------------------------------------------------
    def build_knowledge_base(self, data_dir: str) -> int:
        """建库：索引目录下的文档，然后编译 LangGraph。"""
        self._data_dir = str(data_dir)
        n = self.retriever.index_directory(data_dir)
        if self.long_term is None:
            self.long_term = self._build_long_term(data_dir)
        self._flush_cache()
        self.graph = build_graph(self.cfg, self.retriever)
        return n

    def rebuild_knowledge_base(self, data_dir: str) -> int:
        """重建知识库索引（上传/删除文件后调用，使变更立即生效）。

        做法：丢弃旧的检索器与向量库，用最新的目录内容重新建索引，
        并重新编译图（图持有 retriever 引用，必须重建）。

        注意：长期记忆**不重建**，否则上传一个文件就会把记忆清空。
        """
        self._data_dir = str(data_dir)
        # 1) 新建一个全新的检索器（清空旧向量）
        self.retriever = Retriever(self.cfg.retrieval, self.embeddings, root=self.root)
        # 2) 重新索引目录（命中的嵌入向量直接读磁盘缓存，不重复花钱）
        n = self.retriever.index_directory(data_dir)
        self._flush_cache()
        # 3) 重新编译图，让它指向新的 retriever
        self.graph = build_graph(self.cfg, self.retriever)
        return n

    def _flush_cache(self) -> None:
        if self.embedding_cache is not None:
            self.embedding_cache.flush()

    # ------------------------------------------------------------
    def ask(self, question: str, verbose: bool = True) -> str:
        """回答一个问题（完整走一遍记忆 + 图推理）。"""
        if self.graph is None:
            raise RuntimeError("请先调用 build_knowledge_base() 建库")

        # 1) 组装初始状态：把两类记忆先读出来放进上下文
        initial = {
            "question": question,
            "short_term": self.short_term.as_text(),
            "long_term": self.long_term.recall(question) if self.long_term else [],
            "loops": 0,
            "trace": [],
        }

        # 2) 跑图（同步 invoke；也可换成 .astream 做流式）
        final = self.graph.invoke(initial)
        answer = final.get("answer") or final.get("draft", "")

        # 3) 打印执行轨迹（方便你观察多智能体是怎么协作的）
        if verbose:
            for line in final.get("trace", []):
                print("   ", line)
            if final.get("retrieved"):
                print("    [检索命中]")
                for d in final["retrieved"]:
                    flag = " (邻块扩展)" if d.get("expanded") else ""
                    print(f"      - {d['source']}#{d.get('chunk_id', '?')} (score={d['score']}){flag}")

        # 4) 更新记忆：本轮问答同时写进短期窗口和长期记忆
        self.short_term.add(question, answer)
        if self.long_term is not None:
            self.long_term.add(question, answer)
        return answer

    # ------------------------------------------------------------
    def stats(self) -> dict:
        """运行统计，供 CLI / 前端展示（也方便你观察缓存省了多少调用）。"""
        return {
            "chunks": self.retriever.doc_count,
            "sources": len(self.retriever.sources()),
            "memories": len(self.long_term) if self.long_term else 0,
            "cache_size": self.embedding_cache.size if self.embedding_cache else 0,
            "cache_hits": self.embedding_cache.hits if self.embedding_cache else 0,
            "cache_misses": self.embedding_cache.misses if self.embedding_cache else 0,
            # 实际生效的向量库后端（如配置写 milvus 但环境不支持，
            # 工厂会降级到 numpy——这里如实上报，不粉饰）
            "backend": self.retriever.backend,
            "collection": self.retriever.collection,
            "vectors": self.retriever.vectors,
            "index_type": self.retriever.index_type,
            "metric": self.retriever.metric,
        }
