# ============================================================
# LangGraph 共享状态
# ------------------------------------------------------------
# LangGraph 的核心思想：把"流程"建模成一张图，图上的每个节点是
# 一个函数（Agent），它们共享同一份"状态"。节点读状态 -> 产出更新
# -> 框架按 reducer 合并回状态。这让多智能体协作变得显式、可追踪。
#
# 这里用 TypedDict 描述状态结构；Annotated[..., operator.add] 表示
# 该字段用"追加"方式合并（多节点写入不互相覆盖），这是 LangGraph 的
# 关键机制之一。
# ============================================================
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """在整个图里流转的共享状态。"""

    question: str                                   # 用户当前问题
    route: str                                      # 路由决策："RAG" 或 "DIRECT"
    short_term: str                                 # 短期记忆（最近对话）
    long_term: list[str]                            # 长期记忆召回的历史问答
    retrieved: list[dict]                           # RAG 检索到的文档片段
    draft: str                                      # 生成节点产出的初稿
    critique: str                                   # 批判节点的反馈
    needs_improvement: bool                         # 是否需要再审
    loops: int                                      # 已循环次数（防无限自省）
    answer: str                                     # 最终答案
    trace: Annotated[list[str], operator.add]       # 执行轨迹（追加式合并）
