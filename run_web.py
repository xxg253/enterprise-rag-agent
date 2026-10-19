# ============================================================
# Web 服务启动入口
# ------------------------------------------------------------
# 用法：
#   python run_web.py
# 然后在浏览器打开： http://127.0.0.1:8000
#
# 也可以用 uvicorn 直接起：
#   uvicorn src.web_server:app --reload --port 8000
# ============================================================
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

# 保证项目根目录在 sys.path 中
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    print("=" * 56)
    print("  企业级 RAG 多智能体知识助手 已启动")
    print("  请在浏览器打开： http://127.0.0.1:8000")
    print("  按 Ctrl+C 停止服务")
    print("=" * 56)
    # app 字符串指向 src/web_server.py 里的 FastAPI 实例
    uvicorn.run("src.web_server:app", host="127.0.0.1", port=8000, reload=False)
