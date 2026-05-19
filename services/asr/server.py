"""ASR 模型服务 — faster-whisper (CTranslate2 格式)"""

import base64
import tempfile
import time
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="ASR Service")

_whisper_model = None
_model_ready = False

# CTranslate2 模型目录（包含 model.bin 等文件）
_MODEL_DIR = Path("/models/whisper-base-ct2")
_MODEL_BIN = Path("/models/model.bin")


def _check_whisper_model():
    return _MODEL_DIR.is_dir() or _MODEL_BIN.exists()


def get_whisper_model():
    global _whisper_model, _model_ready
    if _model_ready and _whisper_model:
        return _whisper_model

    if not _check_whisper_model():
        return None

    try:
        from faster_whisper import WhisperModel
        logger.info("加载 whisper 模型...")
        # 优先使用 CTranslate2 目录，否则尝试 model.bin
        model_path = str(_MODEL_DIR) if _MODEL_DIR.is_dir() else str(_MODEL_BIN)
        _whisper_model = WhisperModel(model_path, device="cpu", compute_type="int8")
        _model_ready = True
        logger.info("whisper 模型加载完成")
        return _whisper_model
    except ImportError:
        logger.error("faster-whisper 未安装")
        return None
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return None


class InferRequest(BaseModel):
    input: str
    model: str = "whisper"
    params: dict = None


@app.get("/health")
async def health():
    ready = _check_whisper_model()
    return {
        "status": "ok",
        "models": {
            "whisper": {"weight_ready": ready, "engine": "faster-whisper"},
        }
    }


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()

    if req.model == "whisper":
        model = get_whisper_model()
        if model is None:
            return {
                "output": "[whisper 模型未就绪：权重文件未找到或 faster-whisper 未安装]",
                "model": req.model,
                "latency_ms": int((time.time() - start) * 1000),
            }

        params = req.params or {}
        language = params.get("language", None)

        try:
            audio_bytes = base64.b64decode(req.input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            segments, info = model.transcribe(tmp_path, language=language)
            text = " ".join(seg.text for seg in segments).strip()
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"推理失败: {e}")
            text = f"[识别错误: {str(e)}]"

        return {"output": text, "model": req.model, "latency_ms": int((time.time() - start) * 1000)}

    return {
        "output": f"[模型 {req.model} 暂未实现真实推理]",
        "model": req.model,
        "latency_ms": int((time.time() - start) * 1000),
    }
