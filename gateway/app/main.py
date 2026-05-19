"""FastAPI 入口 — 路由注册 + 依赖注入"""

import asyncio
import logging
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import get_settings
from .adapters.asr import ASRAdapter
from .adapters.translate import TranslateAdapter
from .adapters.ocr import OCRAdapter
from .adapters.tts import TTSAdapter
from .adapters.nlp import NLPAdapter
from .pipelines.registry import get_pipeline, list_pipelines

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local AI Swiss Army",
    description="本地小模型全栈 Demo — 12 模型 / 5 类 / Mock 优先",
    version="0.1.0",
)

# ── 适配器实例 ──

def _init_adapters():
    s = get_settings()
    return {
        "asr": ASRAdapter(s.asr_url, s.local_ai_mock),
        "translate": TranslateAdapter(s.translate_url, s.local_ai_mock, mtran_url=s.mtran_url),
        "ocr": OCRAdapter(s.ocr_url, s.local_ai_mock),
        "tts": TTSAdapter(s.tts_url, s.local_ai_mock),
        "nlp": NLPAdapter(s.nlp_url, s.local_ai_mock),
    }

adapters = _init_adapters()


# ── 模型注册表 ──

MODEL_REGISTRY = {
    "whisper": {
        "name": "faster-whisper-base",
        "category": "asr",
        "desc": "CPU 量化语音识别 (CTranslate2)",
        "files": ["models/asr/whisper-base-ct2/model.bin"],
        "url": "https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
        "size_mb": 148,
    },
    "funasr": {
        "name": "Fun-ASR",
        "category": "asr",
        "desc": "阿里达摩院语音识别",
        "files": ["models/asr/funasr-model"],
        "url": "",
        "size_mb": 350,
    },
    "moonshine": {
        "name": "Moonshine",
        "category": "asr",
        "desc": "超轻量 ASR",
        "files": ["models/asr/moonshine-model"],
        "url": "",
        "size_mb": 50,
    },
    "mtran": {
        "name": "MTranServer",
        "category": "translate",
        "desc": "离线翻译服务（Docker 内置）",
        "files": [],
        "builtin": True,
        "url": "",
        "size_mb": 300,
    },
    "argos": {
        "name": "Argos Translate",
        "category": "translate",
        "desc": "开源离线翻译（Docker 内置）",
        "files": [],
        "builtin": True,
        "url": "",
        "size_mb": 200,
    },
    "ppocr": {
        "name": "PPOCR-v5",
        "category": "ocr",
        "desc": "百度飞桨 OCR（Docker 内置）",
        "files": [],
        "builtin": True,
        "url": "",
        "size_mb": 10,
    },
    "tesseract": {
        "name": "Tesseract",
        "category": "ocr",
        "desc": "经典 OCR 引擎（Docker 内置）",
        "files": [],
        "builtin": True,
        "url": "",
        "size_mb": 30,
    },
    "piper": {
        "name": "Piper",
        "category": "tts",
        "desc": "轻量语音合成",
        "files": ["models/tts/piper-model"],
        "url": "",
        "size_mb": 25,
    },
    "outetts": {
        "name": "OuteTTS-0.6B",
        "category": "tts",
        "desc": "小型神经 TTS",
        "files": ["models/tts/outetts-model"],
        "url": "",
        "size_mb": 600,
    },
    "openaudio": {
        "name": "OpenAudio S1-Mini",
        "category": "tts",
        "desc": "开源语音合成",
        "files": ["models/tts/openaudio-model"],
        "url": "",
        "size_mb": 500,
    },
    "smollm2": {
        "name": "SmolLM2",
        "category": "nlp",
        "desc": "小型语言模型",
        "files": ["models/nlp/smollm2.gguf"],
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf",
        "size_mb": 1050,
    },
    "qwen": {
        "name": "Qwen2.5-0.5B",
        "category": "nlp",
        "desc": "通义千问轻量版",
        "files": ["models/nlp/qwen2.5-0.5b-instruct-q4_k_m.gguf"],
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_mb": 400,
    },
}

# 下载进度追踪
_download_tasks: dict[str, dict] = {}


def _check_model_downloaded(model_id: str) -> bool:
    cfg = MODEL_REGISTRY[model_id]
    # Docker 内置模型，始终就绪
    if cfg.get("builtin"):
        return True
    for f in cfg["files"]:
        p = Path(f)
        if p.exists() and p.stat().st_size > 0:
            return True
    return False


# ── Pydantic 模型 ──

class InferRequest(BaseModel):
    input: str
    model: str = ""
    params: dict = None

class PipelineRequest(BaseModel):
    input: str
    pipeline: str = ""
    params: dict = None

class CustomPipelineRequest(BaseModel):
    input: str
    steps: list[str]
    params: dict = None


# ── 健康检查 ──

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 模型状态 ──

async def _probe_service_health(url: str) -> dict:
    """探测服务的 /health 端点，返回详细模型就绪信息"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


@app.get("/status")
async def status():
    """返回所有服务可用性（含模型权重检测）"""
    result = {}
    for name, adapter in adapters.items():
        available = adapter.is_available()
        models_info = {}

        if available:
            health_data = await _probe_service_health(adapter.service_url)
            models_info = health_data.get("models", {})

        result[name] = {
            "available": available,
            "mock": adapter.should_mock(),
            "url": adapter.service_url,
            "models": models_info,
        }
    return result


# ── 模型管理 API ──

@app.get("/models")
async def list_models():
    """列出所有模型及其下载状态"""
    models = []
    for mid, cfg in MODEL_REGISTRY.items():
        downloaded = _check_model_downloaded(mid)
        progress = _download_tasks.get(mid)
        models.append({
            "id": mid,
            "name": cfg["name"],
            "category": cfg["category"],
            "desc": cfg["desc"],
            "size_mb": cfg["size_mb"],
            "downloaded": downloaded,
            "has_url": bool(cfg["url"]),
            "progress": progress,
        })
    return {"models": models}


@app.post("/models/{model_id}/download")
async def download_model(model_id: str):
    """触发模型下载"""
    if model_id not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_id}"}

    cfg = MODEL_REGISTRY[model_id]
    if not cfg["url"]:
        return {"error": f"模型 {cfg['name']} 暂不支持在线下载，需手动配置"}

    if _check_model_downloaded(model_id):
        return {"message": f"模型 {cfg['name']} 已存在", "already_exists": True}

    if model_id in _download_tasks and _download_tasks[model_id].get("status") == "downloading":
        return {"message": f"模型 {cfg['name']} 正在下载中", "already_downloading": True}

    # 启动后台下载任务
    _download_tasks[model_id] = {
        "status": "downloading",
        "progress_pct": 0,
        "downloaded_mb": 0,
        "total_mb": cfg["size_mb"],
    }
    asyncio.create_task(_do_download(model_id, cfg))

    return {"message": f"开始下载 {cfg['name']}", "started": True}


async def _do_download(model_id: str, cfg: dict):
    """后台下载模型文件"""
    import httpx
    url = cfg["url"].replace("https://huggingface.co/", "https://hf-mirror.com/")
    target = Path(cfg["files"][0])
    target.parent.mkdir(parents=True, exist_ok=True)

    task = _download_tasks[model_id]
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", cfg["size_mb"] * 1024 * 1024))
                downloaded = 0
                with open(target, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        task["progress_pct"] = round(downloaded / total * 100, 1) if total > 0 else 0
                        task["downloaded_mb"] = round(downloaded / 1024 / 1024, 1)
                        task["total_mb"] = round(total / 1024 / 1024, 1)

        task["status"] = "done"
        task["progress_pct"] = 100
        task["downloaded_mb"] = task["total_mb"]
        logger.info(f"模型 {cfg['name']} 下载完成: {target}")
    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        logger.error(f"模型 {cfg['name']} 下载失败: {e}")
        if target.exists():
            target.unlink()


@app.get("/models/{model_id}/progress")
async def model_download_progress(model_id: str):
    """查询下载进度"""
    if model_id not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_id}"}

    downloaded = _check_model_downloaded(model_id)
    progress = _download_tasks.get(model_id)

    if downloaded and (not progress or progress.get("status") != "downloading"):
        return {"status": "done", "progress_pct": 100, "downloaded": True}

    if not progress:
        return {"status": "pending", "progress_pct": 0, "downloaded": downloaded}

    return {**progress, "downloaded": downloaded}


# ── 单模型推理路由 ──

@app.post("/asr")
async def asr_infer(req: InferRequest):
    return await adapters["asr"].infer(req.input, req.model, req.params)

@app.post("/translate")
async def translate_infer(req: InferRequest):
    return await adapters["translate"].infer(req.input, req.model, req.params)

@app.post("/ocr")
async def ocr_infer(req: InferRequest):
    return await adapters["ocr"].infer(req.input, req.model, req.params)

@app.post("/tts")
async def tts_infer(req: InferRequest):
    return await adapters["tts"].infer(req.input, req.model, req.params)

@app.post("/nlp")
async def nlp_infer(req: InferRequest):
    return await adapters["nlp"].infer(req.input, req.model, req.params)


# ── 管线路由 ──

@app.get("/pipelines")
async def list_all_pipelines():
    return {"pipelines": list_pipelines()}

@app.post("/pipeline/voice")
async def voice_pipeline(req: PipelineRequest):
    from .pipelines.voice import voice_pipeline
    return await voice_pipeline(
        audio_input=req.input,
        asr_adapter=adapters["asr"],
        nlp_adapter=adapters["nlp"],
        translate_adapter=adapters["translate"],
        tts_adapter=adapters["tts"],
        params=req.params,
    )

@app.post("/pipeline/document")
async def document_pipeline(req: PipelineRequest):
    from .pipelines.document import document_pipeline
    return await document_pipeline(
        image_input=req.input,
        ocr_adapter=adapters["ocr"],
        nlp_adapter=adapters["nlp"],
        params=req.params,
    )

@app.post("/pipeline/custom")
async def custom_pipeline(req: CustomPipelineRequest):
    """
    自定义管线 — 按 steps 列表顺序依次执行
    steps 格式: ["asr:whisper", "nlp:qwen", "tts:piper"]
    """
    steps = req.steps
    params = req.params or {}
    current_input = req.input
    results = []
    total_start = time.time()

    for step_spec in steps:
        parts = step_spec.split(":")
        svc = parts[0]
        model = parts[1] if len(parts) > 1 else ""

        if svc not in adapters:
            return {"error": f"未知服务: {svc}"}

        result = await adapters[svc].infer(current_input, model=model, params=params.get(svc, {}))
        results.append({"step": svc, "model": model or "default", "result": result})
        current_input = result.get("output", current_input)

    total_ms = int((time.time() - total_start) * 1000)
    return {
        "pipeline": "custom",
        "steps": results,
        "final_output": current_input,
        "total_latency_ms": total_ms,
    }


# ── Web 面板 ──

@app.get("/", response_class=HTMLResponse)
async def web_panel():
    template_path = Path(__file__).parent.parent / "web" / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Web 面板未找到</h1><p>请确保 web/index.html 存在</p>", status_code=404)
