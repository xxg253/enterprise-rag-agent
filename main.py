# ============================================================
# 项目入口（CLI）
# ------------------------------------------------------------
# 用法：
#   python main.py ask "你们的退款政策是什么？"     # 单次提问
#   python main.py chat                              # 交互式多轮对话
#   python main.py demo                              # 一键演示（推荐先跑这个）
#
# 默认配置是 mock 模式，无需任何 Key 就能把整条流水线跑通。
# 要接真实模型：编辑 config.yaml，把 provider 改成 openai 并填 api_key。
# ============================================================
from __future__ import annotations

import sys
from pathlib import Path

import typer

# 让 src 包可被导入（把项目根目录加入 sys.path）
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.assistant import EnterpriseAssistant   # noqa: E402
from src.config import AppConfig                # noqa: E402

app = typer.Typer(add_completion=False, help="企业级 RAG 多智能体知识助手")


def _build_assistant() -> EnterpriseAssistant:
    """按配置构建助手并建立知识库索引。"""
    cfg = AppConfig.load(str(ROOT / "config.yaml"))
    assistant = EnterpriseAssistant(cfg, ROOT)
    n = assistant.build_knowledge_base(str(ROOT / "data"))
    s = assistant.stats()
    print(f"[启动] 已索引 {n} 个文档块（{s['sources']} 个文件），模式={cfg.llm.provider}")
    if assistant.embedding_cache is not None:
        print(f"[缓存] 本次嵌入调用：命中 {s['cache_hits']} 条 / 新算 {s['cache_misses']} 条")
    print(f"[记忆] 长期记忆 {s['memories']} 条\n")
    return assistant


@app.command()
def ask(question: str):
    """单次提问。"""
    assistant = _build_assistant()
    print(f"用户: {question}")
    answer = assistant.ask(question)
    print(f"助手: {answer}\n")


@app.command()
def demo():
    """一键演示：连续几轮提问，展示 RAG + 记忆 + 多智能体协作。"""
    assistant = _build_assistant()
    questions = [
        "你好呀",                                  # 会走 DIRECT 分支（路由能力）
        "我们的产品支持哪些退款方式？",             # 会走 RAG 分支（检索知识库）
        "刚说的那个退款，处理要多久？",             # 依赖短期记忆理解"那个退款"
    ]
    for q in questions:
        print("=" * 60)
        print(f"用户: {q}")
        answer = assistant.ask(q)
        print(f"助手: {answer}\n")


@app.command()
def chat():
    """交互式多轮对话（输入 exit 退出）。"""
    assistant = _build_assistant()
    print("进入对话模式，输入 exit 退出。\n")
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "退出"):
            break
        if not q:
            continue
        answer = assistant.ask(q, verbose=False)
        print(f"助手: {answer}\n")


if __name__ == "__main__":
    app()
