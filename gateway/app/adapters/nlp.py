"""NLP 适配器 — SmolLM2 / Qwen2.5-0.5B（via llama.cpp）"""

from .base import BaseServiceAdapter
from ..mock import mock_nlp

SUPPORTED_MODELS = ["smollm2", "qwen"]


class NLPAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto"):
        super().__init__(service_url, "NLP", mock_mode)

    async def infer(self, input_data: str, model: str = "qwen", params: dict = None) -> dict:
        params = params or {}
        task = params.get("task", "chat")

        if model not in SUPPORTED_MODELS:
            return {"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"}

        if self.should_mock():
            return mock_nlp(model, task)

        payload = {
            "input": input_data,
            "model": model,
            "params": params,
        }
        return await self.call_service(payload)
