"""OCR 模型服务 — PPOCR-tiny / Tesseract"""

import base64
import tempfile
import time
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="OCR Service")

_ppocr_instance = None


def _get_ppocr():
    global _ppocr_instance
    if _ppocr_instance is not None:
        return _ppocr_instance
    from paddleocr import PaddleOCR
    logger.info("初始化 PaddleOCR 实例...")
    _ppocr_instance = PaddleOCR(lang="ch")
    logger.info("PaddleOCR 实例初始化完成")
    return _ppocr_instance


class InferRequest(BaseModel):
    input: str
    model: str = "ppocr"
    params: dict = None


@app.get("/health")
async def health():
    models = {}
    try:
        import pytesseract
        models["tesseract"] = {"weight_ready": True, "engine": "pytesseract"}
    except ImportError:
        models["tesseract"] = {"weight_ready": False}
    try:
        import paddleocr
        models["ppocr"] = {"weight_ready": True, "engine": "paddleocr"}
    except ImportError:
        models["ppocr"] = {"weight_ready": False}
    return {"status": "ok", "models": models}


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()

    if req.model == "ppocr":
        try:
            ocr = _get_ppocr()
            img_bytes = base64.b64decode(req.input)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(img_bytes)
                tmp_path = f.name
            result = ocr.ocr(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            # PaddleOCR v3 返回列表，每个元素为字典，文本在 rec_texts 字段
            texts = []
            if result and isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and "rec_texts" in item:
                        texts.extend(item["rec_texts"])
            output = "\n".join(texts)
            return {"output": output, "model": "ppocr", "latency_ms": int((time.time() - start) * 1000)}
        except ImportError:
            return {"output": "[PPOCR 未安装: pip install paddleocr paddlepaddle]", "model": req.model, "latency_ms": int((time.time() - start) * 1000)}
        except Exception as e:
            return {"output": f"[OCR 错误: {str(e)}]", "model": req.model, "latency_ms": int((time.time() - start) * 1000)}

    if req.model == "tesseract":
        try:
            import pytesseract
            from PIL import Image
            import io
            img_bytes = base64.b64decode(req.input)
            image = Image.open(io.BytesIO(img_bytes))
            params = req.params or {}
            lang = params.get("lang", "chi_sim+eng")
            output = pytesseract.image_to_string(image, lang=lang)
            return {"output": output.strip(), "model": "tesseract", "latency_ms": int((time.time() - start) * 1000)}
        except ImportError:
            return {"output": "[Tesseract 未安装]", "model": req.model, "latency_ms": int((time.time() - start) * 1000)}
        except Exception as e:
            return {"output": f"[OCR 错误: {str(e)}]", "model": req.model, "latency_ms": int((time.time() - start) * 1000)}

    return {"output": f"[未知模型: {req.model}]", "model": req.model, "latency_ms": int((time.time() - start) * 1000)}
