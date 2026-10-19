# ============================================================
# 向量库（Vector Store）
# ------------------------------------------------------------
# 这是 RAG 的"记忆体"：把每个文档块的向量存起来，检索时用
# 用户问题的向量去比"谁最像"（余弦相似度 / 点积）。
#
# 本项目以 **Milvus** 为默认向量库（生产级、持久化、亿级可扩展）：
#   - MilvusVectorStore：默认后端。走 Milvus，支持 HNSW / IVF 索引、
#     标量过滤、数据持久化，重启服务不需要重建索引。
#   - NumpyVectorStore：零依赖降级兜底。Milvus 不可用时保证项目仍能跑通，
#     同时它的实现足够简单，便于在面试里讲清"余弦相似度检索的原理"。
# 二者接口完全一致（同一套 add / search），这就是"面向接口编程"：
# 换向量库不动业务代码。
#
# 【Milvus 设计要点】
#   1) 主键用**内容哈希**（sha1(source|chunk_id|text)）而不是自增 ID：
#      这样重复索引同一批文档时走 upsert 覆盖，不会产生重复数据，
#      索引重建天然幂等；
#   2) 向量入库前做 L2 归一化，metric_type 用 IP（内积）——
#      归一化之后内积就等于余弦相似度，这是向量检索的标准做法；
#   3) 文本原文、来源、块号都作为标量字段存进 Milvus，
#      所以检索结果可以直接从库里还原成 Document，不需要在内存里留副本。
# ============================================================
from __future__ import annotations

import hashlib
from pathlib import Path

from typing import Protocol

import numpy as np

from langchain_core.embeddings import Embeddings
from src.rag.loader import Document


class VectorStore(Protocol):
    """向量库接口协议：任何实现只要满足这两个方法即可被替换。"""

    def add(self, documents: list[Document]) -> None: ...

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]: ...


# ------------------------------------------------------------
# 公共工具：向量归一化
# ------------------------------------------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。

    为什么：余弦相似度 = 归一化之后的内积。这样向量库只需要算 IP（内积），
    就能得到与余弦完全等价的排序结果，索引和查询都省一次计算。
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # 避免零向量除零（空文本或哈希碰撞时可能出现全 0 向量）
    norms[norms == 0.0] = 1.0
    return x / norms


class NumpyVectorStore:
    """基于 numpy 的极简向量库（零依赖，作为 Milvus 不可用时的降级兜底）。

    原理一目了然：
      1) add：把所有文档块编码成向量，堆成一个 (N, D) 矩阵
      2) search：把 query 也编码成向量，与矩阵做点积（向量已归一化，点积=余弦相似度），
         取分数最高的 top_k 个。
    它没有倒排索引/量化压缩，也没有持久化，数据量大时会慢——
    这正是 Milvus 存在的意义。面试时可以用它讲原理，用 Milvus 讲生产。
    """

    backend = "numpy"

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self._docs: list[Document] = []
        self._matrix: np.ndarray | None = None  # 所有块向量的矩阵

    @property
    def docs(self) -> list[Document]:
        """已入库的全部文档块（检索层的邻块扩展需要用到）。"""
        return self._docs

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        vecs = np.array(
            self.embeddings.embed_documents([d.content for d in documents]),
            dtype=np.float32,
        )
        if vecs.ndim == 1:          # 只有一条时保证是 (1, D)
            vecs = vecs.reshape(1, -1)
        # 追加而不是覆盖：多次 add() 也能保持"文档 <-> 向量行号"一一对应
        # （旧版这里是 `self._matrix = np.array(vecs)` 覆盖式写入，
        #   只要 add() 被调用两次就会行号错位，检索张冠李戴）
        if self._matrix is None or len(self._matrix) == 0:
            self._matrix = vecs
        else:
            self._matrix = np.vstack([self._matrix, vecs])
        self._docs.extend(documents)

    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        if self._matrix is None or len(self._docs) == 0 or top_k <= 0:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)  # (D,)
        # 点积得到每个块的相似度分数（向量已归一化 => 内积即余弦）
        scores = self._matrix @ q
        # argsort 升序，取最后 top_k 个再反转 => 降序
        n = min(top_k, len(self._docs))
        idx = np.argsort(scores)[::-1][:n]
        return [(self._docs[i], float(scores[i])) for i in idx]


class MilvusVectorStore:
    """基于 Milvus 的生产级向量库（默认后端，与 numpy 版同接口）。

    连接两种形态，由 uri 决定：
      - uri 以 .db 结尾 / 设为 "local"：**Milvus Lite**，
        本地嵌入式文件库，零部署（注意：官方版仅支持 Linux / macOS）；
      - uri 是 http://host:19530：连接 **Milvus Standalone / Zilliz Cloud**，
        这是生产环境的标准用法。

    schema 设计（为什么这么设计）：
      pk         VARCHAR，主键 = 内容哈希，幂等 upsert 的基础
      vector     FLOAT_VECTOR，向量字段，建 HNSW 索引
      text       VARCHAR，文档块原文，检索结果直接还原成 Document
      source     VARCHAR，来源文件名，支撑"按来源过滤"与溯源
      chunk_id   INT64，块序号，支撑邻块扩展与前端定位

    归一化 + IP：入库与查询都做 L2 归一化，内积等价于余弦相似度。
    """

    backend = "milvus"

    def __init__(
        self,
        embeddings: Embeddings,
        uri: str = "milvus.db",
        collection: str = "rag_chunks",
        index_type: str = "HNSW",
        metric: str = "IP",
        search_params: dict | None = None,
    ):
        self.embeddings = embeddings
        self.uri = uri
        self.collection = collection
        self.index_type = index_type
        self.metric = metric
        self.search_params = search_params or {"ef": 64}

        # 延迟导入：没装 pymilvus 时不影响 numpy 版使用
        try:
            from pymilvus import MilvusClient
        except ImportError as e:      # pragma: no cover - 环境相关
            raise RuntimeError(
                "未安装 pymilvus，无法使用 Milvus 向量库；"
                "请执行 pip install pymilvus，或把 retrieval.vectorstore 改回 numpy"
            ) from e

        # 本地模式必须先建好数据目录，否则 milvus-lite 会抛
        # ConnectionConfigException: Open local milvus failed, dir: ... not exists。
        # 踩坑记录：目录是 milvus-lite 首次启动时自动创建的，但如果目录压根不存在，
        # 它不会自建而是直接失败 —— 于是"新人 clone 下来第一次跑"必然降级成 numpy，
        # 而本机因为目录早就建好了所以完全看不出问题。故这里显式 mkdir(parents=True)。
        if isinstance(uri, str) and not uri.startswith(("http://", "https://", "unix:")):
            try:
                Path(uri).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass  # 建不了就交给下面的连接去报错，走既有降级逻辑

        # uri 规范化：local 关键字 -> 具体的本地 db 文件
        self.client = MilvusClient(uri=str(uri))
        self._ready = False      # collection 是否已建好
        self._loaded = False     # collection 是否已 load 到内存（Milvus 检索前置条件）

        # 已有 collection 就直接接管（持久化的意义：重启不用重建）
        if self.client.has_collection(collection_name=self.collection):
            self._ready = True
            self._ensure_loaded()
            print(f"[milvus] 连接成功，复用已有集合 {self.collection}")
        else:
            print(f"[milvus] 连接成功（{self.uri}），集合 {self.collection} 尚未创建")

    # ------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """把集合 load 到内存后才可检索。

        Milvus 的集合有两态：released（只存盘）/ loaded（载入内存可查）。
        首次 upsert 会自动 load，但**跨进程重新打开一个已存在的集合时**
        它会回到 released 态，直接 search 会报
        "Collection is in state 'released'; call load() before search"。
        这个坑只有真跑一次才会遇到，所以在这里显式兜住。
        """
        if self._loaded or not self._ready:
            return
        try:
            self.client.load_collection(collection_name=self.collection)
        except Exception as e:
            # 部分部署（如 Milvus Lite）会自动加载，加载失败也不该阻断检索
            msg = str(e)
            # 常见于 Windows：Milvus Lite 会在数据目录下写很深的文件，
            # 路径超过 MAX_PATH(260) 就报 "could not open ... for writing"。
            # 这不是权限或数据损坏，而是路径太长 —— 提示用户把项目放在更短的位置。
            if "could not open" in msg and "for writing" in msg:
                print("[milvus] 提示：数据目录路径过长（Windows MAX_PATH 260 字符限制），"
                      "建议把项目移到更短的路径，例如 E:\\rag")
            else:
                print(f"[milvus] load_collection 跳过：{type(e).__name__}: {e}")
        self._loaded = True

    # ------------------------------------------------------------
    # 内部：建表 + 建索引
    # ------------------------------------------------------------
    def _ensure_collection(self, dim: int) -> None:
        """首次写入时按当前向量维度建 collection 与索引。

        为什么把建表放在 add 里：向量维度只有真正调用嵌入模型后才确定，
        构造阶段拿不到。Milvus 侧 collection 一旦建好就会持久化，
        后续进程直接复用，不会重复建。
        """
        from pymilvus import DataType

        schema = self.client.create_schema(
            auto_id=False, enable_dynamic_field=False
        )
        # 注意：pymilvus 3.x 把 add_field 的第二个参数从 dtype 改名成 datatype，
        # 这里一律用**位置参数**传（2.x / 3.x 都接受），避免版本差异导致报错
        schema.add_field(
            "pk", DataType.VARCHAR,
            max_length=64, is_primary=True, auto_id=False,
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_id", DataType.INT64)

        self.client.create_collection(
            collection_name=self.collection, schema=schema
        )

        # 建向量索引：HNSW 在召回率与查询延迟之间最均衡，是默认首选；
        # 数据量特别大时 IVF_FLAT 更省内存，可通过配置切换。
        index_params = self.client.prepare_index_params()
        if self.index_type.upper() == "IVF_FLAT":
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type=self.metric,
                params={"nlist": 128},
            )
        else:
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type=self.metric,
                params={"M": 16, "efConstruction": 200},
            )
        self.client.create_index(
            collection_name=self.collection, index_params=index_params
        )
        self._ready = True
        self._ensure_loaded()
        print(f"[milvus] 已创建集合 {self.collection}（dim={dim}, index={self.index_type}）")

    # ------------------------------------------------------------
    @staticmethod
    def _pk(source: str, chunk_id: int, text: str) -> str:
        """内容哈希主键：同一份内容永远算出同一个 pk。

        这样索引重建时 upsert 会覆盖而不是追加，
        重启服务 / 重新上传文件都不会在库里堆出重复数据。
        """
        raw = f"{source}|{chunk_id}|{text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        vecs = np.array(
            self.embeddings.embed_documents([d.content for d in documents]),
            dtype=np.float32,
        )
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        vecs = l2_normalize(vecs)          # 归一化后 IP == 余弦

        if not self._ready:
            self._ensure_collection(vecs.shape[1])
        else:
            self._ensure_loaded()

        rows = [
            {
                "pk": self._pk(d.source, d.chunk_id, d.content),
                "vector": vecs[i].tolist(),
                "text": d.content,
                "source": d.source,
                "chunk_id": int(d.chunk_id),
            }
            for i, d in enumerate(documents)
        ]
        # 用 upsert 而非 insert：主键是内容哈希，重复索引是幂等覆盖操作
        self.client.upsert(collection_name=self.collection, data=rows)

    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        if not self._ready or top_k <= 0:
            return []
        self._ensure_loaded()
        q = l2_normalize(
            np.array(self.embeddings.embed_query(query), dtype=np.float32)
        ).reshape(1, -1)

        results = self.client.search(
            collection_name=self.collection,
            data=q.tolist(),
            limit=top_k,
            output_fields=["text", "source", "chunk_id"],
            search_params={"metric_type": self.metric, "params": self.search_params},
        )
        out: list[tuple[Document, float]] = []
        for hits in results or []:
            for hit in hits:
                ent = hit.get("entity", {})
                # 文本/来源都在库里，检索结果可以直接还原成 Document，
                # 内存里不需要再留一份副本
                out.append((
                    Document(
                        content=ent.get("text", ""),
                        source=ent.get("source", ""),
                        chunk_id=int(ent.get("chunk_id", -1)),
                    ),
                    float(hit.get("distance", 0.0)),
                ))
        return out

    def __len__(self) -> int:
        """库里实际有多少条向量（来自 Milvus 自身的统计，不是内存计数）。"""
        if not self._ready:
            return 0
        try:
            self._ensure_loaded()
            stats = self.client.get_collection_stats(self.collection)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def reset(self) -> None:
        """清空集合（重新索引前调用）。"""
        try:
            self.client.drop_collection(collection_name=self.collection)
        except Exception:
            pass
        self._ready = False


def build_vectorstore(
    kind: str,
    embeddings: Embeddings,
    retrieval_cfg=None,
    root=None,
) -> VectorStore:
    """向量库工厂：按配置选择实现，失败时优雅降级。

    设计取舍：优先用 Milvus（生产级、可持久化），但**任何异常都降级到 numpy**，
    保证"环境没装 / 没起服务时项目依然能跑"——这是演示项目的底线，
    也是生产系统该有的降级思维。
    """
    kind = (kind or "numpy").lower()
    if kind == "milvus":
        try:
            from pathlib import Path

            uri = getattr(retrieval_cfg, "milvus_uri", "") or "milvus.db"
            if uri in ("local", "lite", ""):
                uri = "milvus.db"
            uri = str(uri)
            if not uri.startswith(("http://", "https://", "unix:")) and root:
                # 本地文件路径统一挂到项目目录下，避免"从哪启动就存哪"
                p = Path(uri)
                if not p.is_absolute():
                    uri = str(Path(root) / p)
            return MilvusVectorStore(
                embeddings,
                uri=uri,
                collection=getattr(retrieval_cfg, "collection_name", "rag_chunks"),
                index_type=getattr(retrieval_cfg, "index_type", "HNSW"),
                metric=getattr(retrieval_cfg, "metric", "IP"),
            )
        except Exception as e:
            print(f"[vectorstore] Milvus 不可用（{type(e).__name__}: {e}），降级为 numpy")
            return NumpyVectorStore(embeddings)
    return NumpyVectorStore(embeddings)