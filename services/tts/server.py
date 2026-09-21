"""TTS 模型服务 — Piper / OuteTTS / OpenAudio"""

import asyncio
import base64
import io
import os
import time
import logging
import wave
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="TTS Service")

# docker-compose 将 ./models/tts 挂载到 /models
_MODEL_DIR = Path("/models")

# Piper 的附加资源下载目录（如拼音方案的 g2pW 注音模型）。
# 必须指向挂载卷，否则容器重建后会重新下载约 150MB。
_DOWNLOAD_DIR = _MODEL_DIR

# 默认语音；拼音方案语音（xiao_ya）需 g2pW 支持，缺失时会尝试自动下载
_DEFAULT_VOICE = os.environ.get("TTS_PIPER_VOICE", "").strip()

# 已加载语音缓存 {voice_id: PiperVoice}，避免每次请求重载模型
_piper_voices: dict = {}


def _discover_voices() -> dict:
    """扫描 Piper 语音：要求 .onnx 与同名 .onnx.json 成对存在"""
    voices = {}
    for onnx_path in sorted(_MODEL_DIR.rglob("*.onnx")):
        if Path(f"{onnx_path}.json").exists():
            voices[onnx_path.stem] = onnx_path
    return voices


def _resolve_voice(model: str, params: dict):
    """解析请求对应的语音（voice_id, onnx_path）；未指定则取第一个可用语音"""
    voices = _discover_voices()
    if not voices:
        return None, None

    # params.voice 优先；model 直接写语音 id（如 zh_CN-huayan-medium）也可
    wanted = params.get("voice") or (model if model != "piper" else None)
    if wanted:
        for voice_id, path in voices.items():
            if voice_id.lower() == str(wanted).lower():
                return voice_id, path
        return None, None

    # 环境变量指定的默认语音优先于扫描顺序
    if _DEFAULT_VOICE:
        for voice_id, path in voices.items():
            if voice_id.lower() == _DEFAULT_VOICE.lower():
                return voice_id, path

    voice_id = next(iter(voices))
    return voice_id, voices[voice_id]


def _get_voice(voice_id: str, onnx_path: Path):
    """加载并缓存 PiperVoice（首次约 1s，之后复用）"""
    if voice_id in _piper_voices:
        return _piper_voices[voice_id]

    from piper import PiperVoice

    logger.info(f"加载 Piper 语音: {voice_id}")
    # download_dir 指向挂载卷：拼音方案语音的 g2pW 注音模型会持久化到宿主机
    voice = PiperVoice.load(str(onnx_path), download_dir=_DOWNLOAD_DIR)
    _piper_voices[voice_id] = voice
    logger.info(f"Piper 语音加载完成: {voice_id}")
    return voice


def _synth_config(voice, params: dict):
    """按需构造 SynthesisConfig —— Piper 各参数默认 None，不传即用模型默认值"""
    from piper import SynthesisConfig

    kwargs = {}
    speaker = params.get("speaker", params.get("speaker_id"))
    if speaker is not None:
        speaker = int(speaker)
        # 单说话人模型传 speaker_id 会报错，越界时忽略
        if getattr(voice.config, "num_speakers", 1) > speaker:
            kwargs["speaker_id"] = speaker
    for key in ("length_scale", "noise_scale", "noise_w_scale", "volume"):
        if params.get(key) is not None:
            kwargs[key] = float(params[key])
    if params.get("normalize_audio") is not None:
        kwargs["normalize_audio"] = bool(params["normalize_audio"])
    return SynthesisConfig(**kwargs) if kwargs else None


def _synthesize_wav(voice, text: str, params: dict) -> bytes:
    """合成完整 WAV 字节（阻塞调用，由 to_thread 承载）"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=_synth_config(voice, params))
    return buf.getvalue()


class InferRequest(BaseModel):
    input: str
    model: str = "piper"
    params: dict = None


@app.get("/health")
async def health():
    voices = _discover_voices()
    return {
        "status": "ok",
        "models": {
            "piper": {
                "weight_ready": bool(voices),
                "engine": "piper-tts",
                "voices": sorted(voices),
                "loaded": sorted(_piper_voices),
                "default_voice": _DEFAULT_VOICE or None,
            },
            "outetts": {"weight_ready": False},
            "openaudio": {"weight_ready": False},
        },
    }


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()

    def elapsed():
        return int((time.time() - start) * 1000)

    if req.model == "piper":
        params = req.params or {}
        voice_id, onnx_path = _resolve_voice(req.model, params)
        if onnx_path is None:
            # 失败一律走 error 字段：返回文本 + format=wav 会被前端当成 base64 音频塞进播放器
            return {
                "error": "Piper 语音未找到：请将 <name>.onnx 与同名 <name>.onnx.json 放入 models/tts/",
                "model": req.model,
                "latency_ms": elapsed(),
            }
        try:
            voice = _get_voice(voice_id, onnx_path)
            audio = await asyncio.to_thread(_synthesize_wav, voice, req.input, params)
        except ImportError:
            logger.error("piper-tts 未安装")
            return {"error": "piper-tts 未安装", "model": req.model, "latency_ms": elapsed()}
        except Exception as e:
            logger.error(f"Piper 合成失败: {e}")
            return {"error": f"TTS 错误: {e}", "model": req.model, "latency_ms": elapsed()}

        return {
            "output": base64.b64encode(audio).decode(),
            "model": "piper",
            "voice": voice_id,
            "latency_ms": elapsed(),
            "format": "wav",
        }

    # OuteTTS / OpenAudio 尚未实现
    return {
        "error": f"TTS 模型 {req.model} 尚未实现真实推理",
        "model": req.model,
        "latency_ms": elapsed(),
    }
