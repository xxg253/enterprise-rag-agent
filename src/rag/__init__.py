# RAG 子包：负责"知识库"的全生命周期
#   loader    -> 读文档
#   splitter  -> 切块
#   vectorstore -> 建索引 + 相似度检索
#   retriever -> 对外统一检索入口（供 Agent 调用）
