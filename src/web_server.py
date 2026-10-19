# ============================================================
# Web 后端（FastAPI）
# ------------------------------------------------------------
# 把已有的 EnterpriseAssistant（RAG + 记忆 + LangGraph）
# 包装成 HTTP 接口，供浏览器前端调用。
#
# 接口设计：
#   GET  /                -> 返回前端页面（index.html）
#   POST /api/chat        -> 普通问答，返回 JSON
#   POST /api/chat/stream -> 流式问答（SSE），边生成边推送
#   GET  /api/health      -> 健康检查
#   GET  /api/info        -> 返回当前配置（模型/知识库块数）供前端展示
#
# 为什么要"流式"：大模型逐字产出，用 SSE 推送能显著改善
# 用户感知速度，这是企业级对话产品的标配交互。
# ============================================================
from __future__ import annotations

import json
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 把项目根目录加入 sys.path，保证能 import src
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.assistant import EnterpriseAssistant   # noqa: E402
from src.config import AppConfig                # noqa: E402

# 全局单例助手（进程启动时构建一次，复用知识库索引）
_cfg: AppConfig | None = None
_assistant: EnterpriseAssistant | None = None
_init_error: str | None = None   # 记录初始化失败原因，供前端友好提示


def get_assistant() -> EnterpriseAssistant:
    """惰性构建助手：第一次调用时才建索引（启动阶段会主动调用一次）。

    若初始化失败（如 API Key 无效、网络不通），会把原因记到 _init_error，
    后续请求抛出带具体原因的异常，而不是含糊的"知识库未初始化"。
    """
    global _cfg, _assistant, _init_error
    if _assistant is None:
        try:
            _cfg = AppConfig.load(str(ROOT / "config.yaml"))
            _assistant = EnterpriseAssistant(_cfg, ROOT)
            n = _assistant.build_knowledge_base(str(ROOT / "data"))
            print(f"[Web] 知识库已索引 {n} 个文档块，模式={_cfg.llm.provider}")
            _init_error = None
        except Exception as e:
            _assistant = None
            _init_error = f"{type(e).__name__}: {e}"
            print(f"[Web] 初始化失败：{_init_error}")
            raise HTTPException(status_code=500, detail=f"初始化失败：{_init_error}")
    return _assistant


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """服务启动时就把知识库索引建好，而不是等第一次提问才建。

    为什么改：过去是"首次请求惰性建索引"，用户第一次提问要干等十几秒，
    并且一次性打出十几次嵌入请求（这正是"第一次问就花 26k token"的来源）。
    挪到启动阶段后：提问响应更快、成本更可预期；
    再配合嵌入磁盘缓存，重启服务建索引几乎不花 token。
    """
    print("=" * 56)
    print("  正在初始化知识库索引…")
    try:
        a = get_assistant()
        s = a.stats()
        print(f"  完成：{s['chunks']} 个块 / {s['sources']} 个文件 / "
              f"长期记忆 {s['memories']} 条 / 嵌入缓存 {s['cache_size']} 条")
        if a.embedding_cache is not None:
            print(f"  本次嵌入调用：命中缓存 {s['cache_hits']} 条，"
                  f"新算 {s['cache_misses']} 条")
    except Exception as e:
        print(f"  启动建索引失败（将在首次请求时重试）：{e}")
    print("=" * 56)
    yield


app = FastAPI(title="企业级 RAG 多智能体知识助手", lifespan=lifespan)


class ChatRequest(BaseModel):
    """前端发来的请求体。"""
    question: str


# ------------------------------------------------------------
# 文件管理相关工具
# ------------------------------------------------------------
DATA_DIR = ROOT / "data"
# 允许上传的扩展名直接复用 loader 里的"受支持格式"，
# 从根上避免"能上传却索引不了"的不一致 bug。
from src.rag.loader import SUPPORTED_EXTS as ALLOWED_EXTS  # noqa: E402


def _safe_filename(name: str) -> str:
    """清洗文件名，防止路径穿越（如 ../../etc/passwd）。

    只保留纯文件名部分，并把危险字符替换掉。
    """
    # 取最后一个路径分隔符之后的部分，丢掉任何目录成分
    base = Path(name).name
    # 替换掉可能引起问题的字符
    for ch in ("/", "\\", "..", "\x00"):
        base = base.replace(ch, "_")
    return base.strip()


def _rebuild() -> int:
    """重建知识库索引，返回索引到的块数。"""
    a = get_assistant()
    return a.rebuild_knowledge_base(str(DATA_DIR))


@app.get("/api/files")
def list_files():
    """列出 data/ 目录下的所有知识库文件。"""
    DATA_DIR.mkdir(exist_ok=True)
    files = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file():
            stat = p.stat()
            files.append({
                "name": p.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "ext": p.suffix.lower(),
                "supported": p.suffix.lower() in ALLOWED_EXTS,   # 是否可被索引
            })
    return {"files": files, "count": len(files)}


@app.get("/api/files/{filename}/preview")
def preview_file(filename: str):
    """预览文件内容（前端"查看"按钮用）。

    返回解析后的纯文本（截断到前 5000 字，避免超大文件卡住浏览器）。
    """
    name = _safe_filename(filename)
    target = DATA_DIR / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在：{name}")

    from src.rag.loader import _PARSERS
    ext = target.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise HTTPException(status_code=400, detail=f"不支持的格式：{ext}")
    try:
        text = parser(target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")

    truncated = len(text) > 5000
    return {
        "name": name,
        "ext": ext,
        "content": text[:5000],
        "truncated": truncated,
        "total_chars": len(text),
    }


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传一个或多个文件到知识库，并重建索引。

    返回里会报告每个文件被索引出的块数，块数为 0 说明内容解析为空
    （例如扫描版 PDF 无可提取文本），前端会给出提示。
    """
    DATA_DIR.mkdir(exist_ok=True)
    saved, skipped = [], []
    for f in files:
        name = _safe_filename(f.filename or "")
        ext = Path(name).suffix.lower()
        if not name:
            continue
        if ext not in ALLOWED_EXTS:
            skipped.append({"name": name, "reason": f"不支持的扩展名 {ext}"})
            continue
        target = DATA_DIR / name
        # 保存文件（流式写入，避免大文件占内存）
        with open(target, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)

    # 重建索引，并统计每个新文件产出的块数（用于识别"解析为空"）
    details = []
    chunks = None
    if saved:
        a = get_assistant()
        chunks = a.rebuild_knowledge_base(str(DATA_DIR))
        # 统计每个上传文件被切成的块数
        from src.rag.loader import load_and_split
        docs = load_and_split(
            str(DATA_DIR),
            a.cfg.retrieval.chunk_size,
            a.cfg.retrieval.chunk_overlap,
            getattr(a.cfg.retrieval, "min_chunk_file", 2000),
        )
        counts: dict[str, int] = {}
        for d in docs:
            counts[d.source] = counts.get(d.source, 0) + 1
        for nm in saved:
            n = counts.get(nm, 0)
            details.append({
                "name": nm,
                "chunks": n,
                "warning": "" if n > 0 else "未能提取到文本内容（可能是扫描版 PDF 或空文件）",
            })

    return {"saved": saved, "skipped": skipped, "indexed_chunks": chunks, "details": details}


@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    """删除知识库中的某个文件，并重建索引。"""
    name = _safe_filename(filename)
    target = DATA_DIR / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在：{name}")
    target.unlink()
    chunks = _rebuild()
    return {"deleted": name, "indexed_chunks": chunks}


@app.get("/api/health")
def health():
    """健康检查接口。"""
    return {"status": "ok"}


@app.get("/api/info")
def info():
    """返回当前运行信息，供前端顶部状态栏展示。

    即使知识库初始化失败，也返回配置信息（附 error 字段），
    这样前端状态栏能显示"模式"和具体错误，而不是笼统的"后端未连接"。
    """
    try:
        get_assistant()
    except HTTPException:
        pass   # 初始化失败不阻断，继续返回配置（含 error）
    cfg = AppConfig.load(str(ROOT / "config.yaml"))
    stats = _assistant.stats() if _assistant is not None else {}
    return {
        "mode": cfg.llm.provider,           # mock 或 openai
        "model": cfg.llm.model,
        # backend 是**实际生效**的后端：配置里写了 milvus，若环境不支持
        # 工厂会降级到 numpy，这里如实上报，前端也照实显示，不粉饰
        "vectorstore": stats.get("backend", cfg.retrieval.vectorstore),
        "vectorstore_cfg": cfg.retrieval.vectorstore,
        "collection": stats.get("collection", ""),
        # Milvus 侧的索引细节：面试演示时能一眼看到"用的什么索引、什么度量"
        "index_type": stats.get("index_type", ""),
        "metric": stats.get("metric", ""),
        "vectors": stats.get("vectors", 0),
        "chunk_size": cfg.retrieval.chunk_size,
        "top_k": cfg.retrieval.top_k,
        "error": _init_error or "",         # 初始化失败原因（正常为空）
        # 运行统计：块数 / 文件数 / 长期记忆条数 / 嵌入缓存命中情况
        "chunks": stats.get("chunks", 0),
        "sources": stats.get("sources", 0),
        "memories": stats.get("memories", 0),
        "cache_size": stats.get("cache_size", 0),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """普通问答：等待完整答案后一次性返回。"""
    a = get_assistant()
    # 关闭 verbose，避免往服务端日志狂刷轨迹
    answer = a.ask(req.question, verbose=False)
    final_trace = []  # 这里为简洁不返回轨迹，流式接口会带
    return {"answer": answer, "trace": final_trace}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答：用 SSE 逐段推送"思考轨迹"和"最终答案"。

    前端用 EventSource / fetch 读取，能看到：
      data: {"type":"trace","content":"[路由] 决策=RAG"}
      data: {"type":"trace","content":"[检索] 召回 4 个片段..."}
      data: {"type":"answer","content":"最终答案..."}
      data: {"type":"done"}
    """
    a = get_assistant()

    async def event_gen() -> AsyncGenerator[str, None]:
        # 注意：这里为兼容同步的图执行，简单地把轨迹与答案分两步推送。
        # 真实生产可用 graph.astream_events 做逐 token 流式。
        try:
            # 1) 先跑完整流程（拿到轨迹与答案）
            import asyncio
            loop = asyncio.get_event_loop()
            # 在线程池里跑同步逻辑，避免阻塞事件循环
            result = await loop.run_in_executor(None, lambda: _run_once(a, req.question))

            for line in result["trace"]:
                yield f"data: {json.dumps({'type': 'trace', 'content': line}, ensure_ascii=False)}\n\n"
            # 把答案按字符分块推送，模拟流式打字效果
            answer = result["answer"]
            step = max(1, len(answer) // 20)
            for i in range(0, len(answer), step):
                yield f"data: {json.dumps({'type': 'answer', 'content': answer[i:i+step]}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:  # 把异常也推给前端，便于排查
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _run_once(a: EnterpriseAssistant, question: str) -> dict:
    """执行一次完整流程，捕获轨迹与答案（供流式接口复用）。"""
    if a.graph is None:
        raise RuntimeError("知识库未初始化")
    initial = {
        "question": question,
        "short_term": a.short_term.as_text(),
        "long_term": a.long_term.recall(question),
        "loops": 0,
        "trace": [],
    }
    final = a.graph.invoke(initial)
    answer = final.get("answer") or final.get("draft", "")
    # 更新记忆
    a.short_term.add(question, answer)
    a.long_term.add(question, answer)

    trace = list(final.get("trace", []))
    # 把向量库的检索明细也推给前端：
    # 让"用了 Milvus"这件事从配置文字变成看得见的召回结果与相似度分数
    for d in final.get("retrieved", []):
        tag = "邻块扩展" if d.get("expanded") else "直接命中"
        trace.append(
            f"     · {d['source']} 块#{d.get('chunk_id', '?')}  "
            f"相似度={d.get('score', 0):.3f}  [{tag}]"
        )
    return {"trace": trace, "answer": answer}


# ------------------------------------------------------------
# 静态前端：把 web/ 目录挂载为静态资源
# ------------------------------------------------------------
WEB_DIR = Path(__file__).parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端首页。"""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
