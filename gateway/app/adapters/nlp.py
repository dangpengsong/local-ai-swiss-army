"""NLP 适配器 — Qwen3-4B

两种后端由 NLP_BACKEND 选择，默认与改动前完全一致：
  llamacpp — 自建服务的 /infer（llama-cpp-python 进程内推理，CPU）
  openai   — 官方 llama.cpp server 的 /v1/chat/completions（CUDA 加速，见 docker-compose.gpu.yml）
两者的 /health 都是 200 表示就绪，基类的探测与降级逻辑无需区分。
"""

import asyncio
import json
import time

import httpx
import logging

from .base import BaseServiceAdapter
from ..mock import mock_nlp

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["qwen3"]

# 与 services/nlp/server.py 的同名常量保持一致。
# 模板与采样参数在 gateway 侧展开，而不是留在服务端：GPU 后端的官方 llama.cpp
# server 没有 task 这个概念，而 CPU / GPU 两种部署必须产出同样的 prompt 和采样
# 结果，所以定义只能有一份权威来源（服务端保留一份仅作直连时的回退）。
SYSTEM_PROMPT = "你是一个有用的AI助手，请用中文回答。"
TASK_PROMPTS = {
    "chat": "{input}",
    "summarize": "请用简洁的中文总结以下内容：\n\n{input}\n\n摘要：",
    "classify": "请对以下文本进行分类，只返回分类名称：\n\n{input}\n\n分类：",
}
MAX_TOKENS = 512
TEMPERATURE = 0.7


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


class NLPAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto", backend: str = "llamacpp"):
        super().__init__(service_url, "NLP", mock_mode)
        self.backend = backend

    def _with_messages(self, input_data: str, params: dict) -> dict:
        """把 task 模板展开成 messages 一并放进 params。

        两种后端都从 params.messages 取用；调用方已经给了就不覆盖。
        """
        if params.get("messages"):
            return params
        template = TASK_PROMPTS.get(params.get("task", "chat"), TASK_PROMPTS["chat"])
        return {
            **params,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": template.format(input=input_data)},
            ],
        }

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

        params = self._with_messages(input_data, params)

        if self.backend == "openai":
            return await self._infer_openai(model, params)

        payload = {
            "input": input_data,
            "model": model,
            "params": params,
        }
        return await self.call_service(payload)

    async def _infer_openai(self, model: str, params: dict) -> dict:
        """官方 llama.cpp server（OpenAI 兼容端点）

        只挑它认识的字段发过去：params 里的 task 等本项目的自定义键它不认，
        一起发虽不报错，但没必要。
        """
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.service_url}/v1/chat/completions",
                    json={
                        "messages": params["messages"],
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            return {
                "output": content.strip(),
                "model": model,
                "mock": False,
                "latency_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            logger.error(f"llama.cpp server 调用失败: {e}")
            # 走 error 字段：带 output 返回会被前端当作"真实推理结果"显示绿色徽章
            return {
                "error": f"llama.cpp server 调用失败: {e}",
                "model": model,
                "latency_ms": int((time.time() - start) * 1000),
            }

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

        params = self._with_messages(input_data, params)

        if self.backend == "openai":
            async for frame in self._stream_openai(model, params):
                yield frame
            return

        payload = {"input": input_data, "model": model, "params": params}
        async for frame in self.stream_service(payload):
            yield frame

    async def _stream_openai(self, model: str, params: dict):
        """把官方 server 的 OpenAI 格式 SSE 转成本项目的 {"delta": ...} 帧"""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.service_url}/v1/chat/completions",
                    json={
                        "messages": params["messages"],
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        # 流里夹杂空行和注释行，只认 data: 帧
                        if not line.startswith("data: "):
                            continue
                        body = line[6:]
                        if body == "[DONE]":
                            break
                        try:
                            delta = (
                                json.loads(body)["choices"][0]
                                .get("delta", {})
                                .get("content")
                            )
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta:
                            yield _sse({"delta": delta})
            yield _sse({
                "done": True,
                "model": model,
                "latency_ms": int((time.time() - start) * 1000),
            })
        except Exception as e:
            logger.error(f"llama.cpp server 流式调用失败: {e}")
            yield _sse({"error": f"llama.cpp server 调用失败: {e}"})
