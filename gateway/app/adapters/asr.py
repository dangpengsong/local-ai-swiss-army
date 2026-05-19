"""ASR 适配器 — whisper.cpp / Fun-ASR / Moonshine"""

from .base import BaseServiceAdapter
from ..mock import mock_asr

SUPPORTED_MODELS = ["whisper", "funasr", "moonshine"]


class ASRAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto"):
        super().__init__(service_url, "ASR", mock_mode)

    async def infer(self, input_data: str, model: str = "whisper", params: dict = None) -> dict:
        params = params or {}

        if model not in SUPPORTED_MODELS:
            return {"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"}

        if self.should_mock():
            return mock_asr(model)

        payload = {
            "input": input_data,
            "model": model,
            "params": params,
        }
        return await self.call_service(payload)
