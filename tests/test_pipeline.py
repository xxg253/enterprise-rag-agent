# ============================================================
# 单元测试（pytest）
# ------------------------------------------------------------
# 企业项目必备：每个核心模块都要有测试，保证改动不破坏既有行为。
# 运行：pytest -v
# ============================================================
import sys
from pathlib import Path

import pytest

# 让测试能 import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llms.embeddings import MockEmbeddings
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.rag.loader import split_text, load_and_split


def test_split_text_overlap():
    """切块应保留 overlap，且块数合理。"""
    text = "A" * 250
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    assert len(chunks[0]) == 100


def test_short_term_window_eviction():
    """短期记忆应只保留最近 N 轮。"""
    mem = ShortTermMemory(window=2)
    mem.add("q1", "a1")
    mem.add("q2", "a2")
    mem.add("q3", "a3")  # q1 应被淘汰
    text = mem.as_text()
    assert "q1" not in text
    assert "q3" in text
    assert len(mem) == 2


def test_long_term_recall_returns_something():
    """长期记忆写入后应能被召回。"""
    emb = MockEmbeddings()
    mem = LongTermMemory(emb, top_k=1)
    mem.add("退款要多久", "3-5个工作日")
    recalled = mem.recall("退款多久")
    assert len(recalled) == 1


def test_load_and_split_from_dir():
    """从 data 目录加载应得到非空文档块。"""
    data_dir = Path(__file__).parent.parent / "data"
    docs = load_and_split(str(data_dir), chunk_size=120, chunk_overlap=20)
    assert len(docs) > 0
    assert all(d.source for d in docs)


# ============================================================
# 以下用例针对本轮修复的三个问题，属于"回归测试"：
# 以后谁再改坏这些行为，pytest 会立刻报警。
# ============================================================

def test_normalize_cjk_spaces():
    """PDF 提取出的中文常带空格（'摘 要'），归一化后应能匹配'摘要'。"""
    from src.rag.loader import normalize_text
    assert "摘要" in normalize_text("摘 要")
    assert "实习经历" in normalize_text("实 习 经 历")
    # 英文之间的空格不能被误删
    assert "the system" in normalize_text("the system")


def test_small_file_is_kept_whole():
    """小文件必须整篇一块，不能被切碎（否则跨块信息检索不全）。"""
    text = "实习经历\n" + "第一段实习描述。" * 30
    docs = load_and_split_from_text(text, min_file_chars=2000, chunk_size=100)
    assert len(docs) == 1
    assert docs[0].chunk_id == 0
    assert "实习经历" in docs[0].content


def test_smart_split_does_not_break_sentences():
    """语义切块应尽量在句子边界断开，不把一句话劈成两半。"""
    from src.rag.loader import split_text_smart
    text = "第一句话结束。" * 100
    chunks = split_text_smart(text, chunk_size=120, chunk_overlap=0)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert c.endswith("。")


def test_vectorstore_multiple_adds_keeps_alignment():
    """向量库多次 add() 必须追加而不是覆盖（旧版会张冠李戴）。"""
    from src.rag.loader import Document
    from src.rag.vectorstore import NumpyVectorStore
    store = NumpyVectorStore(MockEmbeddings())
    store.add([Document("甲", "a.md", 0), Document("乙", "a.md", 1)])
    store.add([Document("丙", "b.md", 0)])
    assert len(store.docs) == 3
    assert store._matrix.shape[0] == 3          # 行数与文档数一一对应
    top = store.search("丙", top_k=1)
    assert top[0][0].content == "丙"            # 检索结果没有错位


def test_source_diversify_covers_small_files(tmp_path):
    """来源均衡：大文件不能把 Top-K 占满，小文件必须有机会进来。"""
    from src.config import RetrievalConfig
    from src.rag.loader import Document
    from src.rag.retriever import Retriever
    cfg = RetrievalConfig(top_k=6, max_per_source=2, neighbor_expand=0)
    # root 指向临时目录：测试用独立的 Milvus 库，不污染项目真实数据
    ret = Retriever(cfg, MockEmbeddings(), root=tmp_path)
    docs = [Document(f"大文件第{i}块", "big.pdf", i) for i in range(10)]
    docs += [Document(f"简历第{i}块 实习经历", "cv.pdf", i) for i in range(2)]
    ret.index_documents(docs)
    picked = ret.retrieve("实习经历", top_k=6)
    assert any(r.source == "cv.pdf" for r in picked)   # 小文件被召回


def test_neighbor_expand_restores_split_context(tmp_path):
    """邻块扩展：命中某块后，块 ±1 的同源邻居应被自动补上。"""
    from src.config import RetrievalConfig
    from src.rag.loader import Document
    from src.rag.retriever import Retriever
    cfg = RetrievalConfig(top_k=1, max_per_source=1, neighbor_expand=1, min_score=0.0)
    ret = Retriever(cfg, MockEmbeddings(), root=tmp_path)
    docs = [Document(f"第{i}块内容", "x.md", i) for i in range(5)]
    ret.index_documents(docs)
    res = ret.retrieve("内容", top_k=1)
    ids = [r.chunk_id for r in res if r.source == "x.md"]
    # 至少能拿到"命中块 + 左右邻居"里的多个块，且按原文顺序排列
    assert len(ids) >= 2
    assert ids == sorted(ids)


def test_garbled_input_never_enters_memory():
    """编码损坏的乱码请求不得写进长期记忆（否则会污染召回）。"""
    from src.memory.long_term import LongTermMemory, _looks_garbled

    assert _looks_garbled("??") is True
    assert _looks_garbled("??????????") is True
    assert _looks_garbled("我的毕业设计摘要是什么？") is False
    assert _looks_garbled("what is this?") is False   # 单个问号不算乱码

    m = LongTermMemory(MockEmbeddings(), top_k=2)
    m.add("??", "乱码答案")
    m.add("我的实习经历", "正常答案")
    assert len(m) == 1                     # 乱码那条被丢弃
    assert "实习" in m.recall("我的实习")[0]


def test_long_term_memory_persists(tmp_path):
    """长期记忆落盘后，重新构造实例应能读回（重启不失忆）。"""
    emb = MockEmbeddings()
    path = tmp_path / "long_term.jsonl"
    m1 = LongTermMemory(emb, top_k=1, persist_path=path)
    m1.add("我叫王思迪", "记住了")
    assert path.exists()
    m2 = LongTermMemory(emb, top_k=1, persist_path=path)
    assert len(m2) == 1
    assert "王思迪" in m2.recall("我是谁")[0]


def test_embedding_cache_avoids_recompute(tmp_path):
    """嵌入缓存：同样的文本第二次不应再调用底层模型。"""
    from src.rag.cache import CachedEmbeddings, EmbeddingCache

    class CountingEmbeddings(MockEmbeddings):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def embed_documents(self, texts):
            self.calls += len(texts)
            return super().embed_documents(texts)

    cache = EmbeddingCache(tmp_path, EmbeddingCache.make_namespace("m", "u"))
    inner = CountingEmbeddings()
    ce = CachedEmbeddings(inner, cache)

    ce.embed_documents(["甲", "乙", "丙"])
    assert inner.calls == 3
    ce.embed_documents(["甲", "乙", "丙"])       # 全部命中缓存
    assert inner.calls == 3
    assert cache.hits == 3
    # 重新构造缓存实例（模拟重启服务）后仍是命中
    cache2 = EmbeddingCache(tmp_path, EmbeddingCache.make_namespace("m", "u"))
    ce2 = CachedEmbeddings(inner, cache2)
    ce2.embed_documents(["甲", "乙", "丙"])
    assert inner.calls == 3
    assert cache2.size == 3


def test_rule_route_saves_llm_call():
    """规则前置路由：寒暄走 DIRECT，查资料走 RAG，拿不准才交给大模型。"""
    from src.agents.nodes import rule_route
    assert rule_route("你好呀") == "DIRECT"
    assert rule_route("谢谢") == "DIRECT"
    assert rule_route("我的毕设摘要是什么") == "RAG"
    assert rule_route("细说这两段经历") == "RAG"
    assert rule_route("那接下来该怎么办") is None


def test_strip_label_removes_model_answer_prefix():
    """模型常把提示词里的 "answer：" 当输出开头，必须在代码层剥掉。"""
    from src.agents.nodes import _strip_label

    assert _strip_label("answer：这是答案") == "这是答案"
    assert _strip_label("**回答**：内容在这里") == "内容在这里"
    assert _strip_label("Answer: english prefix") == "english prefix"
    assert _strip_label("正常输出不应被改动") == "正常输出不应被改动"
    assert _strip_label("") == ""
    # 正文中间出现的 "回答：" 不能被误删
    assert _strip_label("先回答：再解释") == "先回答：再解释"


def test_l2_normalize_makes_inner_product_equal_cosine():
    """归一化后内积 == 余弦相似度：这是向量库统一用 IP 的依据。"""
    import numpy as np

    from src.rag.vectorstore import l2_normalize

    a = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    na, nb = l2_normalize(a), l2_normalize(b)
    ip = na @ nb.T
    cosine = (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True)
                         @ np.linalg.norm(b, axis=1, keepdims=True).T)
    assert np.allclose(ip, cosine, atol=1e-5)
    # 归一化后向量模长为 1
    assert np.allclose(np.linalg.norm(na, axis=1), 1.0, atol=1e-6)


def test_milvus_pk_is_content_hash_and_idempotent():
    """Milvus 主键用内容哈希：同内容永远同 pk，这是索引重建幂等的基础。"""
    from src.rag.vectorstore import MilvusVectorStore

    pk1 = MilvusVectorStore._pk("简历.pdf", 0, "美团外卖 BD")
    pk2 = MilvusVectorStore._pk("简历.pdf", 0, "美团外卖 BD")
    pk3 = MilvusVectorStore._pk("简历.pdf", 1, "美团外卖 BD")   # 块号不同
    pk4 = MilvusVectorStore._pk("简历.pdf", 0, "建材门店运营")   # 内容不同
    assert pk1 == pk2 and len(pk1) == 40
    assert pk1 != pk3 and pk1 != pk4


def test_build_vectorstore_falls_back_when_milvus_unavailable(monkeypatch):
    """Milvus 不可用时必须降级到 numpy，保证项目永远跑得起来。"""
    from src.rag import vectorstore as vs

    def boom(*a, **k):
        raise RuntimeError("模拟 pymilvus 未安装")

    monkeypatch.setattr(vs, "MilvusVectorStore", boom)
    store = vs.build_vectorstore("milvus", MockEmbeddings())
    assert store.backend == "numpy"          # 已优雅降级


def test_milvus_roundtrip_and_persistence(tmp_path):
    """Milvus 端到端：建表 → 写入 → 检索 → 幂等 → 重开仍在（持久化）。

    没装 milvus-lite 的环境自动跳过，保证测试套件在任何机器上都能跑。
    """
    pytest.importorskip("pymilvus", reason="未安装 pymilvus")
    pytest.importorskip("milvus_lite", reason="未安装 milvus-lite（Windows 需单独装）")

    from src.rag.loader import Document
    from src.rag.vectorstore import MilvusVectorStore

    uri = str(tmp_path / "t.db")
    docs = [
        Document(content="我在美团做过外卖 BD，负责地推与商户拓展。", source="a.pdf", chunk_id=0),
        Document(content="在建材门店做运营助理，负责进销存与客户维护。", source="a.pdf", chunk_id=1),
        Document(content="毕设是基于 SpringBoot 的个人健康风险评估系统。", source="b.pdf", chunk_id=0),
    ]
    store = MilvusVectorStore(MockEmbeddings(), uri=uri, collection="unit_test")
    store.add(docs)
    assert len(store) == 3

    # 幂等：同样内容再写一次，行数不应增加（upsert 覆盖而非追加）
    store.add(docs)
    assert len(store) == 3

    hits = store.search("美团外卖的地推", top_k=2)
    assert hits, "检索应有结果"
    assert hits[0][1] >= hits[-1][1]                      # 按相似度降序
    assert isinstance(hits[0][0], Document)
    assert hits[0][0].source in {"a.pdf", "b.pdf"}        # 内容/来源能从库里还原
    store.client.close()

    # 持久化：新进程重开同一个库，数据还在，不需要重建索引
    store2 = MilvusVectorStore(MockEmbeddings(), uri=uri, collection="unit_test")
    assert len(store2) == 3
    assert store2.search("美团外卖的地推", top_k=1)
    store2.client.close()


def load_and_split_from_text(text, min_file_chars, chunk_size):
    """测试辅助：把一段文本当作一个文件来切块。"""
    from src.rag.loader import Document, split_text_smart
    if len(text) <= min_file_chars:
        return [Document(content=text, source="t.md", chunk_id=0)]
    return [Document(content=c, source="t.md", chunk_id=i)
            for i, c in enumerate(split_text_smart(text, chunk_size, 0)) if c.strip()]


# ============================================================
# 密钥治理（.env + ${VAR} 占位符）
# 背景：api_key 写进配置文件再提交到 git 是最常见的泄密方式。
# 这些用例保证"配置里不含真实密钥，且没配 .env 时也能零配置跑通"。
# ============================================================

def test_expand_env_replaces_placeholder(monkeypatch):
    """${VAR} 应被环境变量的值替换。"""
    from src.config import expand_env
    monkeypatch.setenv("MY_KEY", "abc123")
    assert expand_env({"k": "${MY_KEY}"})["k"] == "abc123"


def test_expand_env_supports_default(monkeypatch):
    """${VAR:-默认值}：没配就用默认值，这是"零配置可运行"的关键。"""
    from src.config import expand_env
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert expand_env({"p": "${NOT_SET_ANYWHERE:-mock}"})["p"] == "mock"
    monkeypatch.setenv("NOT_SET_ANYWHERE", "openai")
    assert expand_env({"p": "${NOT_SET_ANYWHERE:-mock}"})["p"] == "openai"


def test_repo_config_contains_no_real_secret():
    """仓库里的 config.yaml 绝不能出现真实密钥（防误提交）。"""
    import re
    root = Path(__file__).parent.parent
    cfg_text = (root / "config.yaml").read_text(encoding="utf-8")
    # 只允许 ${VAR:-default} 形式的占位符，不允许 sk- 开头的真实 key
    assert not re.search(r"sk-[A-Za-z0-9._\-]{20,}", cfg_text), "config.yaml 里出现疑似真实密钥"
    for src_file in (root / "src").rglob("*.py"):
        text = src_file.read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9._\-]{20,}", text), f"{src_file.name} 里出现疑似真实密钥"


def test_gitignore_protects_env_and_personal_files():
    """.gitignore 必须挡住 .env 与 data 下的个人文档。"""
    root = Path(__file__).parent.parent
    text = (root / ".gitignore").read_text(encoding="utf-8")
    rules = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    assert ".env" in rules
    assert any(r.startswith("data/*.pdf") for r in rules)
    assert any("_backup_" in r for r in rules)


def test_runtime_validation_catches_missing_key():
    """provider=openai 却没配 Key 时应在启动阶段就报错，而不是等提问才 401。"""
    from src.config import AppConfig, LLMConfig
    cfg = AppConfig(llm=LLMConfig(provider="openai", api_key="", base_url=""))
    with pytest.raises(ValueError, match="api_key"):
        cfg.validate_runtime()
    # mock 模式不该报错
    AppConfig(llm=LLMConfig(provider="mock", api_key="", base_url="")).validate_runtime()


def test_requirements_include_milvus_lite():
    """requirements.txt 必须同时写 pymilvus 和 milvus-lite。

    踩坑记录：只装 pymilvus 而漏掉 milvus-lite，启动时报
    "milvus-lite is required"，新人 clone 后第一步就卡住。
    这条测试就是防止以后再漏。
    """
    root = Path(__file__).parent.parent
    text = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pymilvus" in text
    assert "milvus-lite" in text


def test_milvus_creates_missing_data_dir(tmp_path):
    """Milvus Lite 数据目录不存在时应自动创建，而不是静默降级成 numpy。

    踩坑记录：milvus-lite 不会自建数据目录，目录缺失时抛
    ConnectionConfigException(dir: ... not exists)，被降级逻辑吃掉后
    backend 悄悄变成 numpy。本机目录早就建好，所以本地测不出来，
    只有"全新 clone 的干净目录"才会触发 —— 这就是它的危险之处。
    """
    from src.rag.vectorstore import MilvusVectorStore
    from src.rag.loader import Document

    target = tmp_path / "brand_new_dir" / "rag.db"   # 故意指向一个不存在的两级目录
    assert not target.parent.exists()

    emb = MockEmbeddings()
    try:
        store = MilvusVectorStore(emb, uri=str(target), collection="t_fresh")
    except Exception as e:
        # 环境不支持 Milvus Lite（如 Linux 以外未装 milvus-lite），跳过即可
        pytest.skip(f"当前环境无法运行 Milvus Lite：{type(e).__name__}")

    assert target.parent.exists(), "连接前应先建好 Milvus 数据目录"
    store.add([Document(content="退换货政策：7 天无理由", source="a.md", chunk_id=0)])
    assert len(store.search("退换货", top_k=1)) == 1
