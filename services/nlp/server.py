"""NLP 模型服务 — SmolLM2 / Qwen2.5-0.5B（via llama-cpp-python）"""

import time
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="NLP Service")

MODEL_FILES = {
    "qwen": "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "smollm2": "/models/smollm2.gguf",
}

_loaded_models: dict = {}


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
        llm = Llama(model_path=MODEL_FILES[model_id], n_ctx=2048, n_threads=4, verbose=False)
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
    model: str = "qwen"
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
