"""文档管线 — OCR → NLP → Output"""

import time
import logging

from ..adapters.ocr import OCRAdapter
from ..adapters.nlp import NLPAdapter
from .registry import register_pipeline

logger = logging.getLogger(__name__)


@register_pipeline("document")
async def document_pipeline(
    image_input: str,
    ocr_adapter: OCRAdapter,
    nlp_adapter: NLPAdapter,
    params: dict = None,
) -> dict:
    """
    文档管线：图片 → OCR 识别 → NLP 摘要/分类
    """
    params = params or {}
    steps = []
    total_start = time.time()

    # Step 1: OCR
    ocr_model = params.get("ocr_model", "ppocr")
    ocr_result = await ocr_adapter.infer(image_input, model=ocr_model)
    steps.append({"step": "ocr", "model": ocr_model, "result": ocr_result})
    text = ocr_result.get("output", "")

    if not text:
        return {"error": "OCR 未识别到文本", "steps": steps}

    # Step 2: NLP
    nlp_model = params.get("nlp_model", "qwen")
    nlp_params = {"task": params.get("nlp_task", "summarize")}
    nlp_result = await nlp_adapter.infer(text, model=nlp_model, params=nlp_params)
    steps.append({"step": "nlp", "model": nlp_model, "result": nlp_result})

    total_ms = int((time.time() - total_start) * 1000)

    return {
        "pipeline": "document",
        "steps": steps,
        "final_output": nlp_result.get("output", ""),
        "total_latency_ms": total_ms,
    }
