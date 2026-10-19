# 多智能体子包（LangGraph 编排）
#   state  -> 全局共享状态（TypedDict），所有节点读写同一份对话上下文
#   nodes  -> 各专职 Agent 节点：路由 / 检索 / 生成 / 批判
#   graph  -> 用 StateGraph 把节点连成带条件边的流程图
