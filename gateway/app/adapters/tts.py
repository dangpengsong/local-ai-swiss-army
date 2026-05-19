"""TTS 适配器 — Piper / OuteTTS / OpenAudio"""

from .base import BaseServiceAdapter
from ..mock import mock_tts

SUPPORTED_MODELS = ["piper", "outetts", "openaudio"]


class TTSAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto"):
        super().__init__(service_url, "TTS", mock_mode)

    async def infer(self, input_data: str, model: str = "piper", params: dict = None) -> dict:
        params = params or {}

        if model not in SUPPORTED_MODELS:
            return {"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"}

        if self.should_mock():
            return mock_tts(model)

        payload = {
            "input": input_data,
            "model": model,
            "params": params,
        }
        return await self.call_service(payload)
