"""NLP 适配器 — Qwen3-4B（llama.cpp server + CUDA）

后端固定为官方 llama.cpp server 的 OpenAI 兼容端点（/v1/chat/completions），
由 docker-compose.yml 里的 nlp 服务提供。本项目是私人定制，文本 AI 只跑 GPU，
所以不再保留 CPU 实现，代码里也就没有后端分支。

工具调用（function calling）的循环放在 gateway 而不是服务端：官方 server
不可能跑我们的循环；而且循环落在这里，将来换任何 OpenAI 兼容后端都不用重写。

一条实测得来的约束：流式 tool_calls 是标准 OpenAI 分片格式 —— 第一片带
id/type/function.name，后续片只有 function.arguments 的碎片，单独看任何一片
都不是合法 JSON，必须按 index 归并、拼完再解析。
"""

import asyncio
import json
import time

import httpx
import logging

from .base import BaseServiceAdapter
from ..mock import mock_nlp
from ..tools import TOOL_SPECS, execute_tool

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["qwen3"]

# 与 GGUF 内嵌的 Qwen3 模板配合使用。模板与采样参数在 gateway 侧展开，
# 而不是留给服务端：官方 llama.cpp server 没有 task 这个概念。
SYSTEM_PROMPT = "你是一个有用的AI助手，请用中文回答。"
TASK_PROMPTS = {
    "chat": "{input}",
    "summarize": "请用简洁的中文总结以下内容：\n\n{input}\n\n摘要：",
    "classify": "请对以下文本进行分类，只返回分类名称：\n\n{input}\n\n分类：",
}
MAX_TOKENS = 512
TEMPERATURE = 0.7

# 工具循环最多几轮。4B 模型偶尔会固执地反复调同一个工具，到顶后改用
# 不带工具的请求强制收尾，避免无限循环。
MAX_TOOL_ROUNDS = 3

# 单轮请求的超时。工具循环把「一个请求一轮推理」变成最多四轮，每轮还要
# prefill 上一轮的 tool 结果，默认超时不够用。
ROUND_TIMEOUT = 300.0


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _merge_tool_call(acc: dict, frag: dict) -> None:
    """累积 tool_calls 分片。

    OpenAI 的分片规则：id 和 function.name 只在第一片出现，function.arguments
    是被切成若干段的 JSON 文本 —— 中途任何一段单独看都不是合法 JSON
    （第一段可能只是个 "{"），所以必须按 index 归并、拼完再解析。
    """
    idx = frag.get("index", 0)
    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    if frag.get("id"):
        slot["id"] = frag["id"]
    fn = frag.get("function") or {}
    if fn.get("name"):
        slot["name"] = fn["name"]
    if fn.get("arguments"):
        slot["arguments"] += fn["arguments"]


def _finalize(acc: dict) -> list:
    """把累积的分片收成完整调用列表，args 解析失败置 None 交给调用方处理"""
    out = []
    for idx in sorted(acc):
        slot = dict(acc[idx])
        raw = slot["arguments"].strip()
        try:
            slot["args"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            slot["args"] = None
        out.append(slot)
    return out


def _normalize_calls(raw: list | None) -> list:
    """归一化非流式的 tool_calls（它本来就是完整的，不需要分片归并）

    arguments 按 OpenAI 标准是字符串；这里也接住 dict，免得服务端版本换了个
    行为就让整个工具功能挂掉。
    """
    out = []
    for i, call in enumerate(raw or []):
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args = raw_args
            raw_args = json.dumps(raw_args, ensure_ascii=False)
        else:
            raw_args = raw_args or ""
            try:
                args = json.loads(raw_args.strip()) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = None
        out.append({
            "id": call.get("id") or f"call_{i}",
            "name": fn.get("name") or "",
            "arguments": raw_args,
            "args": args,
        })
    return out


class NLPAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto",
                 tools_enabled: bool = True):
        super().__init__(service_url, "NLP", mock_mode)
        self.tools_enabled = tools_enabled

    def _with_messages(self, input_data: str, params: dict) -> dict:
        """把 task 模板展开成 messages 一并放进 params；调用方已给就不覆盖"""
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

    def _tools_on(self, params: dict) -> bool:
        """是否为本请求启用工具。管线会显式传 tools=False 关掉。"""
        if not self.tools_enabled:
            return False
        return bool(params.get("tools", True))

    def _payload(self, messages: list, tools: list | None, stream: bool) -> dict:
        body = {
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        if stream:
            body["stream"] = True
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    # ── 非流式 ──

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
        tools_on = self._tools_on(params)
        start = time.time()
        messages = list(params["messages"])
        trace: list = []

        for round_no in range(MAX_TOOL_ROUNDS + 1):
            tools = TOOL_SPECS if (tools_on and round_no < MAX_TOOL_ROUNDS) else None
            try:
                result = await self._call_round(messages, tools)
            except Exception as e:
                logger.error(f"NLP 推理失败: {e}")
                # 走 error 字段：带 output 返回会被前端当作"真实推理结果"显示绿色徽章
                return {
                    "error": f"推理失败: {e}",
                    "model": model,
                    "latency_ms": int((time.time() - start) * 1000),
                }

            calls = result["tool_calls"]
            if not calls:
                out = {
                    "output": result["content"].strip(),
                    "model": model,
                    "mock": False,
                    "latency_ms": int((time.time() - start) * 1000),
                }
                if trace:
                    out["tool_calls"] = trace
                return out

            messages.append(self._assistant_message(result["content"], calls))
            for call, res, ms in await self._run_calls(calls):
                trace.append({
                    "name": call["name"], "args": call["args"],
                    "ok": res["ok"], "detail": res["detail"], "ms": ms,
                })
                messages.append({
                    "role": "tool", "tool_call_id": call["id"], "content": res["content"],
                })

        # MAX_TOOL_ROUNDS + 1 轮之后 tools 恒为 None，模型不可能再要求调用，
        # 所以走不到这里；真走到了说明逻辑有变，如实报错而不是假装成功。
        return {
            "error": "工具调用轮次超出预期",
            "model": model,
            "latency_ms": int((time.time() - start) * 1000),
        }

    async def _call_round(self, messages: list, tools: list | None) -> dict:
        async with httpx.AsyncClient(timeout=ROUND_TIMEOUT) as client:
            resp = await client.post(
                f"{self.service_url}/v1/chat/completions",
                json=self._payload(messages, tools, stream=False),
            )
            resp.raise_for_status()
            data = resp.json()
        message = data["choices"][0]["message"]
        return {
            "content": message.get("content") or "",
            "tool_calls": _normalize_calls(message.get("tool_calls")),
        }

    @staticmethod
    def _assistant_message(content: str, calls: list) -> dict:
        """回填给模型看的 assistant 轮次。

        arguments 用原始字符串（OpenAI 标准，也是官方 server 实测可用的形状），
        不换成解析后的 dict —— 换了模板反而渲染不出来。
        """
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for c in calls
            ],
        }

    @staticmethod
    async def _run_calls(calls: list) -> list:
        """并行执行一轮里的多个工具调用，返回 [(call, result, ms)] 保持原顺序。

        并行是因为一轮里出现多个调用时（「查天气顺便算一下」），串行等待
        会让耗时叠加；工具之间没有依赖，也没有共享状态。
        """
        async def run_one(call: dict):
            began = time.time()
            if call["args"] is None:
                # 通常是模型输出被 max_tokens 截断，参数只写了一半
                res = {
                    "ok": False,
                    "detail": "参数解析失败",
                    "content": f"工具参数不是合法 JSON：{call['arguments'][:200]}",
                }
            else:
                res = await execute_tool(call["name"], call["args"])
            return call, res, int((time.time() - began) * 1000)

        return await asyncio.gather(*(run_one(c) for c in calls))

    # ── 流式 ──

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
        async for frame in self._stream_chat(model, params):
            yield frame

    async def _stream_chat(self, model: str, params: dict):
        """工具循环 + 流式输出的主控

        全程真流式：模型不需要工具时，content 边生成边推，首 token 延迟与
        没有工具的请求完全一致（实测 142ms）。需要工具时，本轮先推完模型
        说的话（通常是「我来查一下」），再执行工具、回填、进入下一轮。
        """
        started = time.time()
        messages = list(params["messages"])
        tools_on = self._tools_on(params)
        trace: list = []

        try:
            for round_no in range(MAX_TOOL_ROUNDS + 1):
                tools = TOOL_SPECS if (tools_on and round_no < MAX_TOOL_ROUNDS) else None
                result = None

                async for kind, payload in self._stream_round(messages, tools):
                    if kind == "delta":
                        yield _sse({"delta": payload})
                    else:
                        result = payload

                calls = result["tool_calls"]
                if not calls:
                    break     # 没有工具调用 → 刚推完的 content 就是最终答案

                # 先广播全部 start 再并行执行：一轮里有多个工具时，前端能同时
                # 看到它们"运行中"，而不是一个一个排队亮起
                for call in calls:
                    yield _sse({"tool": {
                        "id": call["id"], "name": call["name"],
                        "phase": "start", "args": call["args"],
                    }})

                messages.append(self._assistant_message(result["content"], calls))
                for call, res, ms in await self._run_calls(calls):
                    yield _sse({"tool": {
                        "id": call["id"], "phase": "end",
                        "ok": res["ok"], "detail": res["detail"], "ms": ms,
                    }})
                    trace.append({
                        "name": call["name"], "args": call["args"],
                        "ok": res["ok"], "detail": res["detail"], "ms": ms,
                    })
                    # 工具失败也照常回填：实测模型会基于错误信息如实说明，
                    # 而不是编一个结果，所以没有理由中断这一轮
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"], "content": res["content"],
                    })

        except Exception as e:
            logger.error(f"NLP 流式推理失败: {e}")
            yield _sse({"error": f"推理失败: {e}"})
            return

        done = {
            "done": True,
            "model": model,
            "latency_ms": int((time.time() - started) * 1000),
        }
        if trace:
            done["tool_calls"] = trace
        yield _sse(done)

    async def _stream_round(self, messages: list, tools: list | None):
        """一轮流式请求。产出 ("delta", 文本) 流与末尾的 ("result", {...})"""
        acc: dict = {}
        parts: list[str] = []
        async with httpx.AsyncClient(timeout=ROUND_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self.service_url}/v1/chat/completions",
                json=self._payload(messages, tools, stream=True),
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
                        delta = json.loads(body)["choices"][0].get("delta") or {}
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    text = delta.get("content")
                    if text:
                        parts.append(text)
                        yield ("delta", text)
                    for frag in delta.get("tool_calls") or []:
                        _merge_tool_call(acc, frag)
        yield ("result", {"content": "".join(parts), "tool_calls": _finalize(acc)})
