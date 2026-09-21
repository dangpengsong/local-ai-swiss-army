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
from .adapters.tts import TTSAdapter
from .adapters.nlp import NLPAdapter
from .pipelines.registry import get_pipeline, list_pipelines

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local AI Swiss Army",
    description="本地小模型全栈 Demo — 11 模型 / 4 类 / Mock 优先",
    version="0.1.0",
)

# ── 适配器实例 ──

def _init_adapters():
    s = get_settings()
    return {
        "asr": ASRAdapter(s.asr_url, s.local_ai_mock),
        "translate": TranslateAdapter(s.translate_url, s.local_ai_mock, mtran_url=s.mtran_url),
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
        # CTranslate2 目录需这 4 个文件齐全，缺一加载即失败（此前只下 model.bin）
        "files": [
            "models/asr/whisper-base-ct2/model.bin",
            "models/asr/whisper-base-ct2/config.json",
            "models/asr/whisper-base-ct2/tokenizer.json",
            "models/asr/whisper-base-ct2/vocabulary.txt",
        ],
        "urls": [
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/vocabulary.txt",
        ],
        "url": "https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
        "size_mb": 148,
    },
    # planned: 服务端尚未实现，仅在注册表中占位；UI 中标注「未实现」并禁止选择
    "funasr": {
        "name": "Fun-ASR",
        "category": "asr",
        "desc": "阿里达摩院语音识别",
        "files": ["models/asr/funasr-model"],
        "url": "",
        "size_mb": 350,
        "planned": True,
    },
    "moonshine": {
        "name": "Moonshine",
        "category": "asr",
        "desc": "超轻量 ASR",
        "files": ["models/asr/moonshine-model"],
        "url": "",
        "size_mb": 50,
        "planned": True,
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
        "desc": "开源离线翻译（语言包已内置在镜像中）",
        "files": [],
        "builtin": True,
        "url": "",
        # 4 个语言包（en↔zh、en↔ja）合计约 365MB，构建镜像时装好（解压后约占 423MB）
        "size_mb": 365,
    },
    "piper": {
        "name": "Piper 小雅",
        "category": "tts",
        "desc": "轻量语音合成（拼音方案，停顿自然）",
        # Piper 语音需 .onnx 与同名 .onnx.json 成对存在
        "files": [
            "models/tts/piper/zh_CN-xiao_ya-medium.onnx",
            "models/tts/piper/zh_CN-xiao_ya-medium.onnx.json",
        ],
        "urls": [
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx",
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx.json",
        ],
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx",
        "size_mb": 63,
    },
    "piper_huayan": {
        "name": "Piper 华言",
        "category": "tts",
        "desc": "备选中文语音（espeak 方案，中英混读略好）",
        "files": [
            "models/tts/piper/zh_CN-huayan-medium.onnx",
            "models/tts/piper/zh_CN-huayan-medium.onnx.json",
        ],
        "urls": [
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
        ],
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
        "size_mb": 63,
    },
    "outetts": {
        "name": "OuteTTS-0.6B",
        "category": "tts",
        "desc": "小型神经 TTS",
        "files": ["models/tts/outetts-model"],
        "url": "",
        "size_mb": 600,
        "planned": True,
    },
    "openaudio": {
        "name": "OpenAudio S1-Mini",
        "category": "tts",
        "desc": "开源语音合成",
        "files": ["models/tts/openaudio-model"],
        "url": "",
        "size_mb": 500,
        "planned": True,
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


def _check_files_present(model_id: str) -> bool:
    """检查模型文件是否全部下载到位"""
    files = MODEL_REGISTRY[model_id]["files"]
    if not files:
        return False
    # 多文件模型（Piper 的 .onnx + .onnx.json、whisper 的 4 个文件）需全部就位才算就绪
    return all(Path(f).exists() and Path(f).stat().st_size > 0 for f in files)


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


def _model_service_urls(model_id: str) -> list[str]:
    """模型对应的服务地址

    translate 类有两个独立容器：argos 在 translate 服务，mtran 在 mtran 容器。
    """
    s = get_settings()
    return {
        "whisper": [s.asr_url],
        "funasr": [s.asr_url],
        "moonshine": [s.asr_url],
        "argos": [s.translate_url],
        "mtran": [s.mtran_url],
        "piper": [s.tts_url],
        "piper_huayan": [s.tts_url],
        "outetts": [s.tts_url],
        "openaudio": [s.tts_url],
        "smollm2": [s.nlp_url],
        "qwen": [s.nlp_url],
    }.get(model_id, [])


def _service_alive(model_id: str, healths: dict) -> bool:
    """模型所属服务是否在运行"""
    return any((healths.get(u) or {}).get("status") == "ok"
               for u in _model_service_urls(model_id))


async def _probe_urls(urls) -> dict:
    """并发探测多个服务 /health，返回 {url: health_json}

    串行探测时，每个未启动的服务都要等满超时，模型管理页会卡好几秒。
    """
    urls = list(urls)
    if not urls:
        return {}
    healths = await asyncio.gather(*(_probe_service_health(u) for u in urls))
    return dict(zip(urls, healths))


def _builtin_ready(model_id: str, healths: dict) -> bool:
    """builtin 模型的就绪判定 = 对应服务在跑，且该模型在服务里可用

    此前对 builtin 直接 return True，容器没启动时网页仍显示「已就绪」，
    用户据此选中该模型，实际只会拿到 mock 结果。
    """
    for url in _model_service_urls(model_id):
        health = healths.get(url) or {}
        info = (health.get("models") or {}).get(model_id)
        if info is None:
            # 服务健康但未自报模型清单（如 MTranServer 只返回 {"status":"ok"}）→ 以服务存活为准
            if health.get("status") == "ok":
                return True
            continue
        if info.get("weight_ready"):
            return True
    return False


@app.get("/status")
async def status():
    """返回所有服务可用性（含模型权重检测）"""
    async def _one(name: str):
        adapter = adapters[name]
        # any_backend_available 内部是同步 httpx，放线程里跑以免阻塞事件循环
        available = await asyncio.to_thread(adapter.any_backend_available)
        models_info = {}
        if available:
            health_data = await _probe_service_health(adapter.service_url)
            models_info = health_data.get("models", {})
        return name, {
            "available": available,
            "mock": adapter.should_mock(),
            "url": adapter.service_url,
            "models": models_info,
        }

    # 并发探测：串行时每个未启动的服务都要等满超时，状态栏会卡好几秒
    return dict(await asyncio.gather(*(_one(n) for n in adapters)))


# ── 模型管理 API ──

@app.get("/models")
async def list_models():
    """列出所有模型及其下载状态"""
    # 并发探测全部服务（串行会让每个未启动的服务各等满超时）
    urls = {u for mid in MODEL_REGISTRY for u in _model_service_urls(mid)}
    healths = await _probe_urls(urls)

    models = []
    for mid, cfg in MODEL_REGISTRY.items():
        progress = _download_tasks.get(mid)
        running = _service_alive(mid, healths)
        if cfg.get("planned"):
            downloaded = False
        elif cfg.get("builtin"):
            # 无文件可下载，是否就绪全看服务
            downloaded = _builtin_ready(mid, healths)
        else:
            downloaded = _check_files_present(mid)
        models.append({
            "id": mid,
            "name": cfg["name"],
            "category": cfg["category"],
            "desc": cfg["desc"],
            "size_mb": cfg["size_mb"],
            "downloaded": downloaded,
            # 文件齐 ≠ 能用：服务没跑时照样只会返回 mock，前端要能区分
            "service_running": running,
            "builtin": bool(cfg.get("builtin")),
            # planned 模型服务端尚未实现，不能显示成「需手动配置」
            "planned": bool(cfg.get("planned")),
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
    if cfg.get("planned"):
        return {"error": f"模型 {cfg['name']} 尚未实现，暂不可用（计划中）"}
    if cfg.get("builtin"):
        return {"error": f"模型 {cfg['name']} 由 Docker 镜像内置，无需下载，请确认对应服务已启动"}
    if not cfg["url"]:
        return {"error": f"模型 {cfg['name']} 暂不支持在线下载，需手动配置"}

    if _check_files_present(model_id):
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
    """后台下载模型文件（支持 urls 多文件，如 Piper 的 .onnx + .onnx.json）"""
    import httpx
    urls = cfg.get("urls") or [cfg["url"]]
    targets = [Path(f) for f in cfg["files"]]
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)

    def _mirror(url: str) -> str:
        """把 HuggingFace 直链换到 HF_ENDPOINT 指定的源

        此前这里写死 hf-mirror，而 HF_ENDPOINT 只加在 tts 服务上 —— 海外用户
        照 README 改了 .env，面板下载依然全走镜像，与文档承诺的不一致。
        """
        endpoint = get_settings().hf_endpoint.rstrip("/")
        return url.replace("https://huggingface.co/", f"{endpoint}/")

    task = _download_tasks[model_id]
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
            # 预取各文件大小，用于计算整体进度
            sizes = []
            for url in urls:
                try:
                    head = await client.head(_mirror(url))
                    sizes.append(int(head.headers.get("content-length", 0)))
                except Exception:
                    sizes.append(0)
            total = sum(sizes) or cfg["size_mb"] * 1024 * 1024
            task["total_mb"] = round(total / 1024 / 1024, 1)

            downloaded = 0
            for url, target in zip(urls, targets):
                async with client.stream("GET", _mirror(url)) as resp:
                    resp.raise_for_status()
                    with open(target, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            task["progress_pct"] = round(downloaded / total * 100, 1) if total > 0 else 0
                            task["downloaded_mb"] = round(downloaded / 1024 / 1024, 1)

        task["status"] = "done"
        task["progress_pct"] = 100
        task["downloaded_mb"] = task["total_mb"]
        logger.info(f"模型 {cfg['name']} 下载完成: {targets}")
    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        logger.error(f"模型 {cfg['name']} 下载失败: {e}")
        for t in targets:
            if t.exists():
                t.unlink()


@app.get("/models/{model_id}/progress")
async def model_download_progress(model_id: str):
    """查询下载进度"""
    if model_id not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_id}"}

    downloaded = _check_files_present(model_id)
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
