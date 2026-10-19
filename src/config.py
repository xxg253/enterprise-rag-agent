# ============================================================
# 配置加载模块
# ------------------------------------------------------------
# 职责：把 config.yaml 读进来，校验成强类型对象，供全局使用。
#
# 【为什么用 pydantic】
#   配置错误（少字段、类型不对）能在启动阶段就暴露，而不是运行到一半炸掉。
#
# 【密钥为什么不写进代码和配置】
#   api_key 这类凭据写进 config.yaml 再提交到 git，是最常见也最致命的泄密方式——
#   git 历史会永久保留，事后删文件也救不回来。
#   所以这里的约定是：
#     config.yaml 里只写占位符： api_key: ${LLM_API_KEY}
#     真实值放 .env（已被 .gitignore 忽略）或系统环境变量；
#     .env.example 是模板，告诉别人需要配哪些变量。
#   好处：仓库可以公开，别人 clone 下来零配置也能跑（mock 模式），
#   而真实密钥不会离开本机。
# ============================================================
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

# ${VAR} 占位符：加载时用环境变量的值替换
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)((?::-[^}]*)?)\}")


def load_dotenv(path: str | Path) -> None:
    """极简 .env 加载器（只依赖标准库，不引入 python-dotenv）。

    规则：
      - 只在"环境变量里还没有该键"时写入，已有值优先（方便临时覆盖）；
      - 支持 `KEY=value`、`# 注释`、成对引号包裹的值；
      - 找不到文件直接跳过（纯用系统环境变量也能正常运行）。

    为什么不引第三方库：这个项目连向量库都做了降级兜底，
    依赖越少，别人 clone 下来越不容易装不上。
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        # 去掉成对引号（值里可能有 # 或空格）
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value


def _replace_env(match: re.Match) -> str:
    """替换单个 ${VAR} 或 ${VAR:-默认值}。

    语义与 shell / docker-compose 一致：
      ${API_KEY}          取环境变量，没有则为空字符串；
      ${LLM_PROVIDER:-mock}   取环境变量，没有（或为空串）则用 mock。

    有默认值很重要——别人 clone 下来没有 .env 也能零配置跑起来。
    """
    name, fallback = match.group(1), match.group(2)
    value = os.environ.get(name, "")
    if not value and fallback.startswith(":-"):
        return fallback[2:]
    return value


def expand_env(obj):
    """递归把配置里的 ${VAR} / ${VAR:-默认值} 替换成环境变量的值。"""
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(_replace_env, obj)
    return obj


class LLMConfig(BaseModel):
    """大语言模型相关配置。"""

    provider: str = "mock"         # mock | openai
    model: str = "qwen-plus"
    base_url: str = ""             # OpenAI 兼容端点，如 https://xxx/compatible-mode/v1
    api_key: str = ""              # 来自 ${LLM_API_KEY}，不要写死在代码里
    temperature: float = 0.0
    max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    """文本嵌入模型配置（RAG 检索的基石）。"""

    provider: str = "mock"
    model: str = "text-embedding-v3"
    base_url: str = ""
    api_key: str = ""
    # 嵌入向量磁盘缓存：命中的 chunk 不再重复调用接口（省 token 的关键）
    cache: bool = True
    cache_dir: str = "cache"      # 相对项目根目录


class RetrievalConfig(BaseModel):
    """RAG 检索参数。"""

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 6
    vectorstore: str = "milvus"   # milvus（默认，生产级可持久化） | numpy（零依赖兜底）
    # 文件短于该字数则整篇作为一个块（不切），避免"一段经历被劈成两块"
    min_chunk_file: int = 2000
    # 邻块扩展半径：命中某块后，把它前后各 N 个同源块一起带上，
    # 修复"信息被切在块边界、只召回半句话"的问题
    neighbor_expand: int = 1
    # 单一来源最多占 Top-K 里几个名额（防止大文件霸榜挤掉小文件）
    max_per_source: int = 2
    # 相似度下限，低于该分数视为噪声丢弃（0 表示只丢弃负分）
    min_score: float = 0.0

    # ---- Milvus 连接参数（vectorstore=milvus 时生效）----
    # uri 三种写法：
    #   1) "local" 或 "milvus.db"  -> Milvus Lite 本地嵌入式库（免部署）
    #   2) "./milvus/rag.db"       -> 本地文件（相对项目根目录）
    #   3) "http://localhost:19530" -> 连接 Milvus Standalone / Zilliz Cloud
    milvus_uri: str = "milvus.db"
    collection_name: str = "rag_chunks"
    index_type: str = "HNSW"       # HNSW（召回与延迟均衡，默认） | IVF_FLAT（省内存）
    metric: str = "IP"             # IP（内积）+ 归一化 == 余弦相似度


class MemoryConfig(BaseModel):
    """上下文/记忆相关参数。"""

    short_term_window: int = 6
    long_term_top_k: int = 3      # 长期记忆：按语义召回的历史问答条数
    persist: bool = True          # 长期记忆是否落盘（重启后仍记得）
    memory_dir: str = ".memory"   # 相对知识库目录


class GraphConfig(BaseModel):
    """LangGraph 编排参数。"""

    max_critic_loops: int = 1     # "生成 → 批判 → 再审" 最多循环次数
    rule_route_first: bool = True # 路由先用规则判断，命中就省一次大模型调用


class AppConfig(BaseModel):
    """顶层配置，聚合以上所有子配置。"""

    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    memory: MemoryConfig = MemoryConfig()
    graph: GraphConfig = GraphConfig()

    def validate_runtime(self) -> None:
        """启动期自检：配置"看着对但实际跑不起来"的问题，在这里拦下。

        典型场景：provider 写成了 openai 却忘了配 Key。
        过去要等到第一次提问才抛 401，排查成本很高；
        现在启动阶段就明确告诉你是哪个变量没配。
        """
        problems: list[str] = []
        for name, section in (("llm", self.llm), ("embeddings", self.embeddings)):
            if section.provider == "openai":
                if not section.api_key:
                    problems.append(
                        f"{name}.provider=openai 但 api_key 为空（请在 .env 里配置）"
                    )
                if not section.base_url:
                    problems.append(
                        f"{name}.provider=openai 但 base_url 为空（OpenAI 兼容端点地址）"
                    )
        if problems:
            raise ValueError("配置校验失败：\n  - " + "\n  - ".join(problems))

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        """从 YAML 文件加载配置：读 .env → 展开 ${VAR} → pydantic 校验 → 自检。"""
        p = Path(path)
        # .env 与 config.yaml 同级；加载它，密钥就不用进版本库了
        load_dotenv(p.parent / ".env")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # pydantic 会自动把字典映射成强类型对象，缺字段用默认值补
        cfg = cls(**expand_env(data))
        cfg.validate_runtime()
        return cfg