"""语音管线 — ASR → NLP → Translate → TTS"""

import time
import logging

from ..adapters.asr import ASRAdapter
from ..adapters.nlp import NLPAdapter
from ..adapters.translate import TranslateAdapter
from ..adapters.tts import TTSAdapter
from .registry import register_pipeline

logger = logging.getLogger(__name__)


@register_pipeline("voice")
async def voice_pipeline(
    audio_input: str,
    asr_adapter: ASRAdapter,
    nlp_adapter: NLPAdapter,
    translate_adapter: TranslateAdapter,
    tts_adapter: TTSAdapter,
    params: dict = None,
) -> dict:
    """
    语音管线：音频 → ASR 识别 → NLP 理解 → 翻译 → TTS 合成
    """
    params = params or {}
    steps = []
    total_start = time.time()

    # Step 1: ASR
    asr_model = params.get("asr_model", "whisper")
    asr_result = await asr_adapter.infer(audio_input, model=asr_model)
    steps.append({"step": "asr", "model": asr_model, "result": asr_result})
    text = asr_result.get("output", "")

    if not text:
        return {"error": "ASR 未识别到文本", "steps": steps}

    # Step 2: NLP
    nlp_model = params.get("nlp_model", "qwen3")
    nlp_params = {"task": params.get("nlp_task", "chat")}
    nlp_result = await nlp_adapter.infer(text, model=nlp_model, params=nlp_params)
    steps.append({"step": "nlp", "model": nlp_model, "result": nlp_result})
    nlp_text = nlp_result.get("output", text)

    # Step 3: Translate
    translate_model = params.get("translate_model", "mtran")
    # 默认语向是 zh→en 而不是 en→zh：这条管线的输入来自中文语音（内置的中文 TTS 语音、
    # 演示脚本用的都是中文），识别出的中文若按「英译中」送去翻译，只会得到一串乱码。
    # 需要别的语向时用 params 的 translate_source / translate_target 覆盖。
    translate_params = {
        "source": params.get("translate_source", "zh"),
        "target": params.get("translate_target", "en"),
    }
    translate_result = await translate_adapter.infer(
        nlp_text, model=translate_model, params=translate_params
    )
    steps.append({"step": "translate", "model": translate_model, "result": translate_result})
    translated_text = translate_result.get("output", nlp_text)

    # Step 4: TTS
    tts_model = params.get("tts_model", "piper")
    tts_result = await tts_adapter.infer(translated_text, model=tts_model)
    steps.append({"step": "tts", "model": tts_model, "result": tts_result})

    total_ms = int((time.time() - total_start) * 1000)

    return {
        "pipeline": "voice",
        "steps": steps,
        "final_output": tts_result.get("output", ""),
        "format": tts_result.get("format", "wav"),
        "total_latency_ms": total_ms,
    }
