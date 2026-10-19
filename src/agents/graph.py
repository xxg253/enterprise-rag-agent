# ============================================================
# LangGraph 编排（多智能体流程图）
# ------------------------------------------------------------
# 这是整个项目的"大脑接线盘"。我们用 StateGraph 把节点连起来：
#
#        START
#          |
#       [route] ──DIRECT──► [direct] ──► END
#          |
#        RAG
#          ↓
#       [retrieve]
#          ↓
#       [generate] ◄───────┐
#          ↓               │ 需要改进
#       [critique] ────────┘
#          |  不需要改进
#          ▼
#         END
#
# 关键概念：
#   add_node    注册节点（演员）
#   add_edge    固定边（一定要走这条）
#   add_conditional_edges  条件边（根据状态决定走哪条）
#   编译后得到可运行的图，支持 .invoke / .stream 逐步执行
# ============================================================
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    make_critique_node,
    make_generate_node,
    make_retrieve_node,
    make_route_node,
    direct_answer_node,
)
from src.agents.state import AgentState
from src.config import AppConfig
from src.llms.chat import get_chat_model
from src.rag.retriever import Retriever


def build_graph(cfg: AppConfig, retriever: Retriever):
    """构建并编译 LangGraph 多智能体图。

    参数:
        cfg       : 全局配置
        retriever : 已建好索引的 RAG 检索器
    返回:
        编译好的图（可 .invoke(state) 执行）
    """
    llm = get_chat_model(cfg.llm)

    # 1) 实例化各节点
    route_node = make_route_node(llm, cfg.graph.rule_route_first)
    retrieve_node = make_retrieve_node(retriever)
    generate_node = make_generate_node(llm)
    critique_node = make_critique_node(llm, cfg.graph.max_critic_loops)
    direct_node = direct_answer_node(llm)

    # 2) 创建状态图：声明它流转的状态类型是 AgentState
    builder = StateGraph(AgentState)

    # 3) 注册节点
    builder.add_node("route", route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("critique", critique_node)
    builder.add_node("direct", direct_node)

    # 4) 入口：START -> route
    builder.add_edge(START, "route")

    # 5) route 之后的分支：RAG 去检索，DIRECT 去直答
    builder.add_conditional_edges(
        "route",
        lambda s: s.get("route", "RAG"),   # 依据状态里的 route 字段分流
        {"RAG": "retrieve", "DIRECT": "direct"},
    )

    # 6) 检索 -> 生成 -> 审稿（固定边）
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "critique")

    # 7) 审稿之后的条件边：需要改进就回炉重写，否则收尾
    builder.add_conditional_edges(
        "critique",
        lambda s: "rewrite" if s.get("needs_improvement") else "done",
        {"rewrite": "generate", "done": END},   # 回边形成"生成↔审稿"闭环
    )

    # 8) 直答直接结束
    builder.add_edge("direct", END)

    # 9) 编译成可执行图
    return builder.compile()
