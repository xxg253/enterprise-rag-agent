# ============================================================
# 聊天模型层（LLM）
# ------------------------------------------------------------
# 这里展示"依赖注入"的思想：上层 Agent 只依赖一个统一的
# `BaseChatModel` 接口，至于背后是真实大模型还是离线 Mock，
# 完全由配置决定。这也是为什么本项目能"无 Key 跑通整条流水线"——
# 把 LLM 替换成确定性的 Mock，流程照样走，只是内容变成占位。
# ============================================================
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    # 真实模型走 OpenAI 兼容协议（阿里 MaaS / DeepSeek / 混元 都兼容）
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - 极少数环境下未安装
    ChatOpenAI = None

from src.config import LLMConfig


class MockChatModel(BaseChatModel):
    """离线 Mock 聊天模型。

    它并非"真智能"，而是根据提示词里的关键词返回确定性文本，
    目的是让整条 RAG + 多智能体流水线在无网络/无 Key 时也能跑通，
    方便你单步研读每个环节。生产环境把 provider 改成 openai 即可。

    注意点：LangChain 的 BaseChatModel 要求我们实现 _generate，
    框架会自动在其之上提供 .invoke / .ainvoke 等统一调用接口，
    这样它就能无缝接入 LangChain 的链和 LangGraph 的节点。
    """

    # pydantic v2 允许任意类型字段
    model_config = {"arbitrary_types_allowed": True}

    def _llm_type(self) -> str:
        return "mock-chat"

    def _craft(self, messages: list[BaseMessage]) -> str:
        """根据最后一条提示词的内容，给出确定性的"伪回答"。"""
        last = messages[-1].content if messages else ""
        text = str(last).lower()

        # 1) 路由节点：决定走 RAG 还是 DIRECT
        #    注意判断顺序很重要——路由/批判这类"结构化输出"节点必须先匹配，
        #    否则会被后面的通用分支抢先命中。
        if any(k in text for k in ("route", "路由", "classify", "分类")):
            # Mock 无法真正理解语义，这里用"问题长度+寒暄词"做启发式模拟：
            # 寒暄/极短问题走 DIRECT，其余走 RAG。
            import re
            q = last.split("用户问题：")[-1].strip()
            greeting = any(g in q for g in ("你好", "谢谢", "hi", "hello", "嗨"))
            if greeting or len(q) <= 4:
                return '{"route": "DIRECT"}'
            return '{"route": "RAG"}'

        # 2) 批判节点：评估生成质量，决定是否需要再审
        if any(k in text for k in ("critique", "评估", "批判", "打分")):
            return '{"needs_improvement": false, "feedback": "回答准确，无需修改。"}'

        # 3) 生成节点：合成最终答案（这里只是占位，真实模型会真正读上下文）
        if any(k in text for k in ("answer", "生成", "回答", "回答用户")):
            # 直接回答分支（DIRECT）：提示词里没有检索片段
            if "<<doc>>" not in text:
                return "[Mock 直答] 你好！我是企业知识助手，可以帮你查询公司政策、产品 FAQ 等问题。"
            # 简单抽取提示词里 <<DOC>>...<</DOC>> 包裹的检索片段
            docs = []
            import re
            for m in re.findall(r"<<DOC>>(.*?)<</DOC>>", str(last), re.DOTALL):
                docs.append(m.strip()[:80])
            ctx = "；".join(docs[:2]) if docs else "（无检索片段）"
            return (
                f"[Mock 生成] 基于以下检索内容作答：{ctx}。"
                f"结论：该问题已结合 RAG 上下文给出回答。"
            )

        return "[Mock 回复] 这是离线模式的占位回答。"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        content = self._craft(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # LangGraph 异步图会调用 ainvoke，这里直接复用同步实现
        return self._generate(messages, stop, run_manager, **kwargs)


def _offline_token_count(self, text: str) -> int:
    """离线 token 计数：避免 tiktoken 联网下载分词表。

    背景：LangChain 的 ChatOpenAI 默认用 tiktoken 计算 token 数，
    而 tiktoken 首次使用会去 `openaipublic.blob.core.windows.net`
    下载分词表。在离线/被墙的环境里这一步会抛 SSLError，导致
    整个服务初始化失败（表现为"知识库未初始化"）。

    这里用一个粗略但稳定的估算规则替代：中文按 1 字 ≈ 1 token，
    英文按 4 字符 ≈ 1 token。仅用于内部计数/日志，不影响实际调用。
    """
    if not text:
        return 0
    # 统计中文字符数
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def _make_offline_chat_model():
    """动态构造一个"不联网计数"的 ChatOpenAI 子类。

    如果 ChatOpenAI 不可用（未安装 langchain-openai），返回 None。
    """
    if ChatOpenAI is None:
        return None

    class _OfflineTokenChatOpenAI(ChatOpenAI):  # type: ignore[misc, valid-type]
        """重写 token 计数方法，彻底避开 tiktoken 的联网下载。"""

        def get_num_tokens(self, text: str) -> int:
            return _offline_token_count(self, text)

        def get_num_tokens_from_messages(self, messages, **kwargs) -> int:
            total = 0
            for m in messages:
                content = getattr(m, "content", "") or ""
                if isinstance(content, str):
                    total += _offline_token_count(self, content)
            return total or 1

    return _OfflineTokenChatOpenAI


# 模块级缓存，避免重复创建类
_OfflineTokenChatOpenAI = _make_offline_chat_model()


def get_chat_model(cfg: LLMConfig) -> BaseChatModel:
    """聊天模型工厂：按配置返回真实模型或 Mock。

    这是典型的"工厂方法 + 依赖注入"——调用方拿到的永远是
    BaseChatModel，换厂商不需要改任何业务代码。
    """
    if cfg.provider == "openai" and ChatOpenAI is not None:
        return _OfflineTokenChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url or None,   # OpenAI 兼容端点
            api_key=cfg.api_key or "EMPTY",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            # 关键：必须设置超时，否则网络异常时请求会无限挂起，
            # 前端表现为"一直转圈没有响应"（历史踩坑）。
            timeout=60,          # 单次请求最多等 60 秒
            max_retries=1,       # 失败最多重试 1 次，别让用户干等
        )
    # 默认（含 mock / 未配置）返回离线 Mock
    return MockChatModel()
