"""NLP 模型服务 — Qwen3-4B（via llama-cpp-python）"""

import asyncio
import json
import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="NLP Service")

MODEL_FILES = {
    "qwen3": "/models/qwen3-4b-instruct-2507-q4_k_m.gguf",
}


def _default_threads() -> int:
    """默认线程数取物理核心数，不要取逻辑核数

    i5-10400 是 6 核 12 线程：n_threads=12 时两个超线程争抢同一个物理核的执行单元，
    实测从 43 tok/s 崩到 1.5 tok/s（掉了 28 倍）。可用 NLP_N_THREADS 覆盖。
    """
    n = os.cpu_count() or 4
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("cpu cores"):
                return int(line.split(":")[1]) or n
    except Exception:
        pass
    return max(1, n // 2)


N_THREADS = int(os.environ.get("NLP_N_THREADS") or _default_threads())
# 2048 对 4B 模型偏局促。KV cache 约 144KB/token（36 层 / 8 个 KV 头），4096 约占 590MB
N_CTX = int(os.environ.get("NLP_N_CTX") or 4096)

_loaded_models: dict = {}

# Llama 不是线程安全的。流式推理走 executor 线程后，两个并发请求会同时进入同一个
# Llama 实例，导致崩溃或输出串台。CPU 推理本就 ~7 tok/s，并发没有收益，直接串行化。
_infer_lock = asyncio.Lock()


def _model_weight_exists(model_id: str) -> bool:
    path = MODEL_FILES.get(model_id, "")
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 1000


def get_model(model_id: str):
    if model_id in _loaded_models:
        return _loaded_models[model_id]
    if not _model_weight_exists(model_id):
        return None
    try:
        from llama_cpp import Llama
        logger.info(f"加载模型 {model_id}: {MODEL_FILES[model_id]}")
        llm = Llama(model_path=MODEL_FILES[model_id], n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)
        _loaded_models[model_id] = llm
        logger.info(f"模型 {model_id} 加载完成")
        return llm
    except ImportError:
        logger.error("llama-cpp-python 未安装")
        return None
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return None


TASK_PROMPTS = {
    "chat": "{input}",
    "summarize": "请用简洁的中文总结以下内容：\n\n{input}\n\n摘要：",
    "classify": "请对以下文本进行分类，只返回分类名称：\n\n{input}\n\n分类：",
}


class InferRequest(BaseModel):
    input: str
    model: str = "qwen3"
    params: dict = None


@app.get("/health")
async def health():
    models = {}
    for mid, path in MODEL_FILES.items():
        ready = _model_weight_exists(mid)
        loaded = mid in _loaded_models
        models[mid] = {"weight_ready": ready, "loaded": loaded}
    return {"status": "ok", "models": models}


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()
    params = req.params or {}
    task = params.get("task", "chat")

    if not _model_weight_exists(req.model):
        # 失败一律走 error 字段：塞进 output 会被标记为"真实"结果展示
        return {
            "error": f"模型 {req.model} 权重未找到，请先下载到 /models/",
            "model": req.model,
            "latency_ms": int((time.time() - start) * 1000),
        }

    llm = get_model(req.model)
    if llm is None:
        return {
            "error": f"模型 {req.model} 加载失败",
            "model": req.model,
            "latency_ms": int((time.time() - start) * 1000),
        }

    prompt_template = TASK_PROMPTS.get(task, TASK_PROMPTS["chat"])
    user_message = prompt_template.format(input=req.input)

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手，请用中文回答。"},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        output = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"推理失败: {e}")
        return {
            "error": f"推理失败: {e}",
            "model": req.model,
            "latency_ms": int((time.time() - start) * 1000),
        }

    return {"output": output, "model": req.model, "latency_ms": int((time.time() - start) * 1000)}


# ── 流式推理 ──

def _sse(obj: dict) -> str:
    """SSE 帧。json.dumps 会把文本里的换行转义成 \\n 两个字符，
    不会在 data: 行里留下裸换行（裸换行会截断 SSE 帧）"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _run_stream(llm, messages, loop, queue):
    """在 executor 线程里消费阻塞生成器，逐块投递到 asyncio 队列

    create_chat_completion(stream=True) 返回同步生成器，直接在 async 路由里迭代会
    占住事件循环，期间 /health 不响应 —— gateway 每 30 秒探一次（超时 2 秒），
    超时即判定服务不可用，降级 Mock 并让面板变红。
    """
    try:
        for chunk in llm.create_chat_completion(
            messages=messages, max_tokens=512, temperature=0.7, stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
    except Exception as e:
        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)  # 结束哨兵


async def _error_stream(message: str):
    yield _sse({"error": message})


@app.post("/infer/stream")
async def infer_stream(req: InferRequest):
    """流式推理。与 /infer 并存：管线（voice/custom）需要完整文本才能喂给下一步"""
    params = req.params or {}
    task = params.get("task", "chat")
    start = time.time()

    if not _model_weight_exists(req.model):
        return StreamingResponse(
            _error_stream(f"模型 {req.model} 权重未找到，请先下载到 /models/"),
            media_type="text/event-stream",
        )

    llm = get_model(req.model)
    if llm is None:
        return StreamingResponse(
            _error_stream(f"模型 {req.model} 加载失败"),
            media_type="text/event-stream",
        )

    prompt_template = TASK_PROMPTS.get(task, TASK_PROMPTS["chat"])
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，请用中文回答。"},
        {"role": "user", "content": prompt_template.format(input=req.input)},
    ]

    async def gen():
        async with _infer_lock:
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _run_stream, llm, messages, loop, queue)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    kind, value = item
                    if kind == "error":
                        yield _sse({"error": f"推理失败: {value}"})
                        return
                    yield _sse({"delta": value})
                yield _sse({
                    "done": True,
                    "model": req.model,
                    "latency_ms": int((time.time() - start) * 1000),
                })
            finally:
                # 客户端中途断开时生成器被关闭（GeneratorExit），但 executor 里的
                # llama.cpp 仍在生成 —— 它无从感知连接已断。必须等线程真正结束再放锁，
                # 否则下一个请求会并发进入同一个 Llama 实例（非线程安全）。
                # _run_stream 内部已捕获全部异常并经队列投递，future 不会抛。
                await future

    return StreamingResponse(gen(), media_type="text/event-stream")
