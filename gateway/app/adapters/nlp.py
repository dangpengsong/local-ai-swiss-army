"""NLP 适配器 — Qwen3-4B（via llama.cpp）"""

import asyncio
import json

from .base import BaseServiceAdapter
from ..mock import mock_nlp

SUPPORTED_MODELS = ["qwen3"]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


class NLPAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto"):
        super().__init__(service_url, "NLP", mock_mode)

    async def infer(self, input_data: str, model: str = "qwen3", params: dict = None) -> dict:
        params = params or {}
        task = params.get("task", "chat")

        # gateway 的 InferRequest.model 默认是空串，会覆盖签名上的默认值；
        # 不传 model 时落到该服务的首选模型
        model = model or SUPPORTED_MODELS[0]

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

    async def stream(self, input_data: str, model: str = "qwen3", params: dict = None):
        """流式推理，产出 SSE 帧

        错误也走流内：HTTP 200 在第一个 token 时就已经发出，之后无法再改状态码，
        只能推一个 error 事件让前端识别。
        """
        params = params or {}
        task = params.get("task", "chat")

        # 同 infer：空 model 落到首选模型
        model = model or SUPPORTED_MODELS[0]

        if model not in SUPPORTED_MODELS:
            yield _sse({"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"})
            return

        if self.should_mock():
            # mock 也必须逐字吐：否则前端要一直停在 loading 等完整结果，
            # 开了 Mock 反而比非流式更迷惑
            for ch in mock_nlp(model, task)["output"]:
                yield _sse({"delta": ch})
                await asyncio.sleep(0.02)
            yield _sse({"done": True, "mock": True, "model": model})
            return

        payload = {"input": input_data, "model": model, "params": params}
        async for frame in self.stream_service(payload):
            yield frame
