"""翻译模型服务 — Argos Translate（MTranServer 通过独立 Docker 容器提供）"""

import time
import logging

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Translate Service")

_argos_installed = False
try:
    import argostranslate.translate
    _argos_installed = True
except ImportError:
    logger.info("argos-translate 未安装")


def _check_language_pair(source: str, target: str) -> bool:
    """检查 argos 语言对是否已安装"""
    if not _argos_installed:
        return False
    try:
        from argostranslate.translate import get_installed_languages
        for lang in get_installed_languages():
            if lang.code == source:
                for t in lang.translations_from:
                    if t.to_code if hasattr(t, 'to_code') else t.code == target:
                        return True
                break
        return False
    except Exception:
        return False


class InferRequest(BaseModel):
    input: str
    model: str = "argos"
    params: dict = None


@app.get("/health")
async def health():
    models = {}
    if _argos_installed:
        pairs = []
        try:
            from argostranslate.translate import get_installed_languages
            for lang in get_installed_languages():
                for t in lang.translations_from:
                    tc = t.to_code if hasattr(t, 'to_code') else t.code
                    pairs.append(f"{lang.code}->{tc}")
        except Exception:
            pass
        models["argos"] = {"weight_ready": True, "engine": "argos-translate", "pairs": pairs}
    else:
        models["argos"] = {"weight_ready": False}
    models["mtran"] = {"weight_ready": False, "note": "独立 Docker 容器"}
    return {"status": "ok", "models": models}


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()
    params = req.params or {}
    source = params.get("source", "en")
    target = params.get("target", "zh")

    if req.model == "argos" and _argos_installed:
        try:
            from argostranslate.translate import translate
            output = translate(req.input, source, target)
            return {
                "output": output,
                "model": "argos",
                "latency_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            logger.error(f"Argos 翻译失败: {e}")
            hint = f" (语言包 {source}->{target} 可能未安装)" if "NoneType" in str(e) else ""
            return {
                "output": f"[翻译错误: {str(e)}{hint}]",
                "model": req.model,
                "latency_ms": int((time.time() - start) * 1000),
            }

    # MTranServer 走独立容器，这里只处理 argos
    return {
        "output": f"[翻译服务: model={req.model}, {source}->{target}] (模型未安装)",
        "model": req.model,
        "latency_ms": int((time.time() - start) * 1000),
    }
