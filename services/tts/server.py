"""TTS 模型服务 — Piper / OuteTTS / OpenAudio"""

import base64
import time
import logging

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="TTS Service")


class InferRequest(BaseModel):
    input: str
    model: str = "piper"
    params: dict = None


@app.get("/health")
async def health():
    from pathlib import Path
    models = {
        "piper": {"weight_ready": Path("/models/piper-model.onnx").exists()},
        "outetts": {"weight_ready": False},
        "openaudio": {"weight_ready": False},
    }
    return {"status": "ok", "models": models}


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()

    if req.model == "piper":
        try:
            import subprocess
            import tempfile
            from pathlib import Path
            # 使用 piper CLI
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out_path = f.name
            model_path = "/models/piper-model"
            if Path(model_path + ".onnx").exists():
                proc = subprocess.run(
                    ["piper", "--model", model_path + ".onnx", "--output_file", out_path],
                    input=req.input.encode("utf-8"),
                    capture_output=True, timeout=30
                )
                if proc.returncode == 0:
                    with open(out_path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode()
                    Path(out_path).unlink(missing_ok=True)
                    return {"output": audio_b64, "model": "piper", "latency_ms": int((time.time() - start) * 1000), "format": "wav"}
            return {"output": "[Piper 模型文件未找到，请下载 .onnx 模型到 /models/piper-model.onnx]", "model": req.model, "latency_ms": int((time.time() - start) * 1000), "format": "wav"}
        except FileNotFoundError:
            return {"output": "[piper 未安装]", "model": req.model, "latency_ms": int((time.time() - start) * 1000), "format": "wav"}
        except Exception as e:
            return {"output": f"[TTS 错误: {str(e)}]", "model": req.model, "latency_ms": int((time.time() - start) * 1000), "format": "wav"}

    # OuteTTS / OpenAudio 占位
    return {
        "output": f"[TTS: {req.model} 尚未实现真实推理]",
        "model": req.model,
        "latency_ms": int((time.time() - start) * 1000),
        "format": "wav",
    }
