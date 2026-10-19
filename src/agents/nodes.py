# ============================================================
# 各 Agent 节点（LangGraph 的"演员"）
# ------------------------------------------------------------
# 每个节点就是一个函数：读共享状态 -> 干活 -> 返回要合并进状态的字段。
# 我们放五个专职节点，体现"多智能体分工"：
#   route_node    : 路由员  —— 判断该问题是否需要查知识库
#   retrieve_node : 检索员  —— 从 RAG 知识库召回相关片段
#   generate_node : 撰写员  —— 结合记忆 + 检索上下文生成答案
#   critique_node : 审稿员  —— 评估答案质量，决定是否打回重写
#   direct_node   : 直答员  —— 寒暄类问题直接回答
#
# 【本版重点修复】"记忆功能体现不出来"
#   旧版有三个问题，导致记忆形同虚设：
#     1) direct_answer_node 的提示词里**根本没有记忆字段**，
#        所以路由判到 DIRECT 时，模型完全不知道你之前说过什么；
#     2) generate_node 虽然把记忆塞进了提示词，但只笼统写了一句
#        "只依据下面提供的上下文作答"，模型读到大量知识库片段后，
#        自然会忽略"用户之前说过的话"；提示词里也没有说清
#        "用户个人信息"和"文档证据"这两类上下文该如何结合；
#     3) 长期记忆只存在内存里，重启即失忆。
#   对策：记忆在所有分支都注入；提示词拆成「已知用户信息」与
#   「知识库证据」两个明确区块并规定结合方式；长期记忆落盘持久化。
# ============================================================
from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agents.state import AgentState
from src.rag.retriever import Retriever


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

# 模型输出里自带的标签前缀，如 "answer：" "最终回答：" "**回答**："
# 为什么需要清洗：提示词结尾写了"请直接输出最终回答，不要加任何前缀或标签。\n"，
# 模型很容易照抄这个标签当成输出开头，前端就会显示 "answer：..."。
# **提示词约束永远不可靠，必须在代码层再兜一道**。
_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?(?:final\s+answer|answer|答案|回答|最终回答|输出)(?:\*\*)?\s*[:：]\s*",
    re.IGNORECASE,
)


def _strip_label(text: str) -> str:
    """剥掉模型输出开头的 "answer：" 这类标签，只留正文。"""
    if not text:
        return ""
    return _LABEL_RE.sub("", text.strip(), count=1).strip()
def _safe_json(text: str) -> dict:
    """大模型有时会输出 ```json ... ```，这里做容错解析。"""
    text = text.strip()
    if "```" in text:
        # 取第一个代码块内容
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


def _memory_block(state: AgentState) -> str:
    """把短期 + 长期记忆渲染成一段统一的"已知用户信息"区块。

    短期记忆 = 本次会话最近几轮对话（你说的原话都在里面）
    长期记忆 = 历史会话里按语义召回的相关问答
    """
    short = (state.get("short_term") or "").strip()
    long_items = state.get("long_term") or []
    long_text = "\n".join(x for x in long_items if x).strip()

    parts = []
    if short:
        parts.append(f"【本次对话最近几轮】\n{short}")
    if long_text:
        parts.append(f"【更早的相关历史（按语义召回）】\n{long_text}")
    return "\n\n".join(parts) if parts else "（暂无已知信息）"


def _docs_block(state: AgentState) -> str:
    """把检索片段渲染成带来源标注的引用块（保持 <<DOC>> 标记，兼容 Mock 模型）。"""
    lines = []
    for i, d in enumerate(state.get("retrieved", []), 1):
        tag = "（相邻上下文）" if d.get("expanded") else ""
        lines.append(f"<<DOC>>[{i}] 来源《{d['source']}》{tag}\n{d['content']}<</DOC>>")
    return "\n".join(lines)


# ------------------------------------------------------------
# 规则前置路由：能不调大模型就不调（省 token）
# ------------------------------------------------------------
_GREETINGS = ("你好", "您好", "hi", "hello", "嗨", "谢谢", "thanks",
              "在吗", "早上好", "下午好", "晚上好", "再见", "晚安")
_KB_HINTS = ("知识库", "文档", "文件", "资料", "简历", "毕设", "毕业设计", "论文",
             "摘要", "政策", "退款", "faq", "产品", "公司", "规定", "流程",
             "我的", "之前", "上次", "刚才", "实习", "经历", "项目")


def rule_route(question: str) -> str | None:
    """用规则先判一次路由；拿不准返回 None，交给大模型决策。

    收益：寒暄/明显查资料这两类问题不再消耗一次 LLM 调用。
    """
    q = (question or "").strip()
    low = q.lower()
    if not q:
        return "DIRECT"
    has_kb = any(h in low for h in _KB_HINTS)
    # 纯寒暄、没提到任何要查的东西 -> 直答
    if len(q) <= 12 and any(g in low for g in _GREETINGS) and not has_kb:
        return "DIRECT"
    # 明显是在问知识库/自己的资料 -> 走 RAG（RAG 分支同样带记忆）
    if has_kb:
        return "RAG"
    return None


def make_route_node(llm: BaseChatModel, rule_first: bool = True):
    """生成"路由节点"。

    为什么要路由：不是所有问题都需要查知识库。比如"你好""谢谢"这类
    寒暄，直接回答即可，检索反而浪费钱、还可能引入噪声。
    """
    def route_node(state: AgentState) -> dict:
        # 1) 先走廉价规则，命中就不必再花一次大模型调用
        if rule_first:
            decision = rule_route(state["question"])
            if decision is not None:
                return {
                    "route": decision,
                    "trace": [f"[路由] 决策={decision}（规则命中，省一次大模型调用）"],
                }

        # 2) 规则拿不准时才问大模型
        prompt = (
            "你是路由员。判断用户问题是否需要检索企业知识库。\n"
            "只需要检索事实/文档类问题，寒暄类直接回答。\n"
            '请只输出 JSON：{"route": "RAG" 或 "DIRECT"}\n'
            f"用户问题：{state['question']}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        decision = _safe_json(resp.content).get("route", "RAG")
        decision = decision if decision in ("RAG", "DIRECT") else "RAG"
        return {"route": decision, "trace": [f"[路由] 决策={decision}"]}

    return route_node


def make_retrieve_node(retriever: Retriever):
    """生成"检索节点"：调用 RAG 检索器召回文档片段。"""
    def retrieve_node(state: AgentState) -> dict:
        chunks = retriever.retrieve(state["question"])
        retrieved = [
            {
                "content": c.content,
                "source": c.source,
                "score": round(c.score, 4),
                "chunk_id": c.chunk_id,
                "expanded": c.expanded,
            }
            for c in chunks
        ]
        srcs = sorted({c["source"] for c in retrieved})
        expanded = sum(1 for c in retrieved if c["expanded"])
        return {
            "retrieved": retrieved,
            "trace": [
                f"[检索] 召回 {len(retrieved)} 个片段（其中 {expanded} 个由邻块扩展补齐），"
                f"覆盖 {len(srcs)} 个文件：{srcs}"
            ],
        }

    return retrieve_node


def make_generate_node(llm: BaseChatModel):
    """生成"撰写节点"：把 短期记忆 + 长期记忆 + 检索片段 一起喂给模型。

    这就是"上下文工程"的落地：同一份提示词里拼装三类上下文，
    模型才能既懂"刚才聊了什么"，又懂"知识库怎么说的"。

    【关键】提示词把上下文明确分成两个区块，并规定它们的优先级与结合方式，
    否则模型会只顾着读知识库片段，把你之前说过的话当空气。
    """
    def generate_node(state: AgentState) -> dict:
        memory_text = _memory_block(state)
        docs_text = _docs_block(state)

        # 如果被批判过，把反馈也带上，让模型重写
        feedback = state.get("critique", "")
        revise_hint = f"\n\n【上一轮审稿意见（请据此改进）】\n{feedback}" if feedback else ""

        prompt = (
            "你是企业知识助手，需要结合【已知用户信息】和【知识库证据】回答用户问题。\n\n"
            "=== 已知用户信息（用户本人告诉过你的，优先级最高）===\n"
            f"{memory_text}\n\n"
            "=== 知识库证据（文档检索片段）===\n"
            f"{docs_text or '（无）'}\n\n"
            f"=== 用户问题 ===\n{state['question']}"
            f"{revise_hint}\n\n"
            "【回答要求】\n"
            "1. 如果问题与用户本人有关（例如「我的…」「我之前说的…」「我告诉过你的…」），"
            "必须先使用【已知用户信息】作答；\n"
            "2. 知识库证据用于补充细节与事实，两类信息应当结合起来回答，"
            "而不是只挑其中一边；\n"
            "3. 只有当【已知用户信息】和【知识库证据】里确实都没有相关内容时，"
            "才可以说信息不足，并且要说明缺的是哪一部分；\n"
            "4. 不要编造，引用知识库内容时注明来源文件名。\n\n"
            "请直接输出最终回答，不要加任何前缀或标签。\n"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        clean = _strip_label(resp.content)
        return {
            "draft": clean,
            "trace": [f"[生成] 产出草稿 {len(clean)} 字（已注入记忆 {len(memory_text)} 字）"],
        }

    return generate_node


def make_critique_node(llm: BaseChatModel, max_loops: int):
    """生成"审稿节点"：评估草稿是否需要改进，并做循环上限保护。"""
    def critique_node(state: AgentState) -> dict:
        loops = state.get("loops", 0)
        # 保险丝：循环次数到顶就强制放行，避免烧 token 的无限自省
        if loops >= max_loops:
            return {
                "needs_improvement": False,
                "answer": state.get("draft", ""),
                "trace": [f"[审稿] 达到最大循环 {max_loops}，直接采用草稿"],
            }

        prompt = (
            "你是审稿员。评估下面的回答是否准确、完整、无编造。\n"
            "特别注意：如果【已知用户信息】里有与该问题相关的内容，"
            "但回答完全没有用上，应当判定为需要改进。\n"
            '只输出 JSON：{"needs_improvement": true/false, "feedback": "..."}\n'
            f"【已知用户信息】\n{_memory_block(state)}\n\n"
            f"问题：{state['question']}\n回答：{state.get('draft', '')}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        verdict = _safe_json(resp.content)
        needs = bool(verdict.get("needs_improvement", False))
        feedback = verdict.get("feedback", "")

        update: dict = {
            "needs_improvement": needs,
            "critique": feedback,
            "loops": loops + 1,
            "trace": [f"[审稿] needs_improvement={needs}"],
        }
        if not needs:
            update["answer"] = state.get("draft", "")   # 通过则定稿
        return update

    return critique_node


def direct_answer_node(llm: BaseChatModel):
    """在路由为 DIRECT 时使用：跳过检索，但**照样注入记忆**。

    【修复点】旧版这里只传了用户问题，所以寒暄/闲聊时模型完全不记得你，
    用户会觉得"记忆功能没生效"。现在把短期 + 长期记忆一起带上，
    直答分支也具备记忆能力。
    """
    def direct_node(state: AgentState) -> dict:
        prompt = (
            "你是企业知识助手。用户这句话属于寒暄或简单交流，请友好自然地回应。\n"
            "如果【已知用户信息】里有用户的身份、经历、偏好等信息，"
            "请自然地体现你记得他（但不要生硬复述）。\n\n"
            "=== 已知用户信息 ===\n"
            f"{_memory_block(state)}\n\n"
            "=== 用户这句话 ===\n"
            f"{state['question']}\n\n"
            "请直接输出最终回答，不要加任何前缀或标签。\n"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"answer": _strip_label(resp.content), "trace": ["[直答] 跳过检索（已注入记忆）"]}

    return direct_node
