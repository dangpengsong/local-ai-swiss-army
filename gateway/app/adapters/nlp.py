"""NLP 适配器 — Qwen3-4B（llama.cpp server + CUDA）

后端固定为官方 llama.cpp server 的 OpenAI 兼容端点（/v1/chat/completions），
由 docker-compose.yml 里的 nlp 服务提供。本项目是私人定制，文本 AI 只跑 GPU，
所以不再保留 CPU 实现，代码里也就没有后端分支。

工具调用（function calling）的循环放在 gateway 而不是服务端：官方 server
不可能跑我们的循环；而且循环落在这里，将来换任何 OpenAI 兼容后端都不用重写。

对外有两组入口，共用同一个工具循环内核（run_tools）：
  infer / stream —— 项目自有格式，网页面板和管线用
  chat / chat_stream —— OpenAI 格式，Cherry Studio 这类外部客户端接入用
循环只有一份，工具行为才处处一致；分开写迟早漂。

一条实测得来的约束：流式 tool_calls 是标准 OpenAI 分片格式 —— 第一片带
id/type/function.name，后续片只有 function.arguments 的碎片，单独看任何一片
都不是合法 JSON，必须按 index 归并、拼完再解析。
"""

import asyncio
import json
import time
import uuid

import httpx
import logging

from .base import BaseServiceAdapter
from ..mock import mock_nlp
from ..tools import TOOL_SPECS, execute_tool

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["qwen3"]

SYSTEM_PROMPT = "你是一个有用的AI助手，请用中文回答。"

# 单次生成上限。这个值直接决定「长回答会不会被切断」：512 时实测让它写一个
# 解析身份证的 PHP 函数，写到一半就停在 finish_reason=length 上。它是**上限
# 而非目标**，答完即停，所以调高不会让简单问题变慢，只让长回答有机会写完。
# 外部客户端自带的 max_tokens 也按这个值封顶。
MAX_TOKENS = 2048

TEMPERATURE = 0.7

# 工具循环最多几轮。4B 模型偶尔会固执地反复调同一个工具，到顶后改用
# 不带工具的请求强制收尾，避免无限循环。
MAX_TOOL_ROUNDS = 3

# 生成空间小于这个数就不带工具。一次工具调用（函数名 + JSON 参数）至少要
# 几十个 token，空间不够时模型会写出被截断的调用，而 llama.cpp 解析不了
# 会**直接返回 500**（实测 max_tokens=20 时触发，报 "Failed to parse tool
# call arguments as JSON"）。宁可这一步不调工具，也好过整个请求失败。
TOOL_MIN_TOKENS = 128

# 单轮请求的超时。工具循环把「一个请求一轮推理」变成最多四轮，每轮还要
# prefill 上一轮的 tool 结果，默认超时不够用。
ROUND_TIMEOUT = 300.0


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _ensure_system(messages: list) -> list:
    """客户端没给 system 就补一个。

    Qwen3 裸跑（完全没有 system）时对话格式会漂，而外部接入方不一定带 system。
    """
    if any(m.get("role") == "system" for m in messages):
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


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


def _context_error(body: bytes) -> str | None:
    """prompt 本身超出上下文时，给一句用户能照做的话；别的错误返回 None

    llama.cpp 只在 **prompt 本身**超限时拒绝，不管 max_tokens 要多少 ——
    实测 prompt 2513 tokens 配 max_tokens 2048（合计 4561 > 4096）照样正常返回，
    它生成到剩余空间用尽就自己以 finish_reason=length 收尾。所以这里不需要
    「收紧 max_tokens 重试」那套，真正会撞上的场景只剩「输入太长」。

    错误体里直接带着用量，不必自己估算字符数：
        {"error":{"code":400,"message":"request (12008 tokens) exceeds the
         available context size (4096 tokens), try increasing it",
         "type":"exceed_context_size_error","n_prompt_tokens":12008,"n_ctx":4096}}
    """
    try:
        err = json.loads(body).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return None
    if err.get("type") != "exceed_context_size_error":
        return None
    n_prompt = err.get("n_prompt_tokens") or 0
    n_ctx = err.get("n_ctx") or 0
    if n_prompt and n_ctx:
        return f"输入过长：{n_prompt} tokens 已超出模型 {n_ctx} 的上下文上限，请缩短输入"
    return "输入超出模型的上下文上限，请缩短输入"


def _upstream_message(body: bytes, status: int) -> str:
    """把上游错误体转成一句能看懂的话，别再套一层 httpx 的异常文本"""
    try:
        err = json.loads(body)
        msg = (err.get("error") or {}).get("message") or err.get("message")
        if msg:
            return f"上游返回 {status}: {msg}"
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"上游返回 {status}: {body[:200].decode('utf-8', 'replace')}"


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


class NLPAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto",
                 tools_enabled: bool = True):
        super().__init__(service_url, "NLP", mock_mode)
        self.tools_enabled = tools_enabled

    def _with_messages(self, input_data: str, params: dict) -> dict:
        """把输入包成 messages 一并放进 params；调用方已给就不覆盖"""
        if params.get("messages"):
            return params
        return {
            **params,
            "messages": _ensure_system([{"role": "user", "content": input_data}]),
        }

    def _tools_on(self, params: dict) -> bool:
        """是否为本请求启用工具。管线会显式传 tools=False 关掉。"""
        if not self.tools_enabled:
            return False
        return bool(params.get("tools", True))

    def _payload(self, messages: list, tools: list | None, stream: bool,
                 max_tokens: int | None = None,
                 temperature: float | None = None) -> dict:
        body = {
            "messages": messages,
            "max_tokens": max_tokens or MAX_TOKENS,
            "temperature": TEMPERATURE if temperature is None else temperature,
        }
        if stream:
            body["stream"] = True
            # 和 OpenAI 一样，llama.cpp 默认不在流里发用量 —— 不显式要这一帧，
            # 客户端拿到的 token 数会一直是 0
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    # ── 工具循环内核（两个渲染层共用）──

    async def run_tools(self, messages: list, tools_on: bool = True,
                        max_tokens: int | None = None,
                        temperature: float | None = None):
        """跑完整个工具循环，产出结构化事件。

          ("delta", text)                模型输出增量
          ("tool_start", call)           开始执行某个工具
          ("tool_end", (call, res, ms))  工具执行完毕
          ("done", {...})                循环结束，带 content / trace / usage

        messages 会被就地追加（assistant 轮与 tool 结果），调用方传进来的是副本。
        """
        trace: list = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        content = ""
        finish = "stop"

        for round_no in range(MAX_TOOL_ROUNDS + 1):
            # 最后一轮去掉 tools：模型不可能再要求调用，只能基于已有信息作答
            budget = max_tokens or MAX_TOKENS
            tools = TOOL_SPECS if (
                tools_on and round_no < MAX_TOOL_ROUNDS and budget >= TOOL_MIN_TOKENS
            ) else None
            result = None
            async for kind, payload in self._stream_round(
                    messages, tools, max_tokens, temperature):
                if kind == "delta":
                    yield ("delta", payload)
                else:
                    result = payload

            content = result["content"]
            finish = result.get("finish_reason") or "stop"
            if result.get("usage"):
                for k in usage:
                    # 多轮要累加而不是取最后：每轮都重发完整 prompt，那些 prefill
                    # 是实打实算过的，客户端看到的 token 数应当反映真实消耗
                    usage[k] += result["usage"].get(k) or 0

            calls = result["tool_calls"]
            if not calls:
                break

            # 先广播全部 start 再并行执行：一轮里有多个工具时，前端能同时
            # 看到它们"运行中"，而不是一个一个排队亮起
            for call in calls:
                yield ("tool_start", call)

            messages.append(self._assistant_message(content, calls))
            for call, res, ms in await self._run_calls(calls):
                yield ("tool_end", (call, res, ms))
                trace.append({
                    "name": call["name"], "args": call["args"],
                    "ok": res["ok"], "detail": res["detail"], "ms": ms,
                })
                # 工具失败也照常回填：实测模型会基于错误信息如实说明，
                # 而不是编一个结果，所以没有理由中断这一轮
                messages.append({
                    "role": "tool", "tool_call_id": call["id"], "content": res["content"],
                })
        else:
            # 理论到不了这里：最后一轮不带 tools，模型无法再要求调用
            logger.warning("工具循环跑满仍未收敛，按当前内容收尾")

        yield ("done", {
            "content": content, "trace": trace, "usage": usage, "finish_reason": finish,
        })

    async def _stream_round(self, messages: list, tools: list | None,
                            max_tokens: int | None = None,
                            temperature: float | None = None):
        """一轮请求。产出 ("delta", 文本) 流与末尾的 ("result", {...})。

        即使调用方要的是非流式结果也走流式请求：循环里的每一轮都要能边收边发，
        否则就得维护两套解析逻辑。llama.cpp 两种模式的采样结果一致。
        """
        acc: dict = {}
        parts: list[str] = []
        usage = None
        finish = None
        async with httpx.AsyncClient(timeout=ROUND_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self.service_url}/v1/chat/completions",
                json=self._payload(messages, tools, True, max_tokens, temperature),
            ) as resp:
                if resp.status_code != 200:
                    # 不用 raise_for_status：它抛出的是 httpx 的通用文案，
                    # 而 llama.cpp 的错误体里带着「输入多少 token、上限多少」
                    # 这种用户能照做的信息，值得原样传达
                    body = await resp.aread()
                    raise RuntimeError(
                        _context_error(body) or _upstream_message(body, resp.status_code)
                    )
                async for line in resp.aiter_lines():
                    line = line.strip()
                    # 流里夹杂空行和注释行，只认 data: 帧
                    if not line.startswith("data: "):
                        continue
                    body = line[6:]
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    # usage 帧的 choices 是空数组，不能直接取 [0]
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        parts.append(text)
                        yield ("delta", text)
                    for frag in delta.get("tool_calls") or []:
                        _merge_tool_call(acc, frag)
        yield ("result", {
            "content": "".join(parts),
            "tool_calls": _finalize(acc),
            "usage": usage,
            "finish_reason": finish,
        })

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

    # ── 入口 1：项目自有格式（网页面板 + 管线）──

    def _resolve_model(self, model: str) -> str | None:
        """空 model 落到首选模型；不支持则返回 None"""
        model = model or SUPPORTED_MODELS[0]
        return model if model in SUPPORTED_MODELS else None

    async def infer(self, input_data: str, model: str = "qwen3", params: dict = None) -> dict:
        params = params or {}
        model = self._resolve_model(model)
        if model is None:
            return {"error": f"不支持的模型: {params.get('model') or ''}，可选: {SUPPORTED_MODELS}"}

        if self.should_mock():
            return mock_nlp(model)

        params = self._with_messages(input_data, params)
        started = time.time()
        out = None
        try:
            async for kind, payload in self.run_tools(
                    list(params["messages"]), self._tools_on(params)):
                if kind == "done":
                    out = payload
        except Exception as e:
            logger.error(f"NLP 推理失败: {e}")
            # 走 error 字段：带 output 返回会被前端当作"真实推理结果"显示绿色徽章
            return {
                "error": f"推理失败: {e}",
                "model": model,
                "latency_ms": int((time.time() - started) * 1000),
            }

        result = {
            "output": out["content"].strip(),
            "model": model,
            "mock": False,
            "latency_ms": int((time.time() - started) * 1000),
        }
        if out["trace"]:
            result["tool_calls"] = out["trace"]
        return result

    async def stream(self, input_data: str, model: str = "qwen3", params: dict = None):
        """流式推理，产出项目自有 SSE 帧

        错误也走流内：HTTP 200 在第一个 token 时就已经发出，之后无法再改状态码，
        只能推一个 error 事件让前端识别。
        """
        params = params or {}
        model = self._resolve_model(model)
        if model is None:
            yield _sse({"error": f"不支持的模型: {params.get('model') or ''}，可选: {SUPPORTED_MODELS}"})
            return

        if self.should_mock():
            # mock 也必须逐字吐：否则前端要一直停在 loading 等完整结果，
            # 开了 Mock 反而比非流式更迷惑
            for ch in mock_nlp(model)["output"]:
                yield _sse({"delta": ch})
                await asyncio.sleep(0.02)
            yield _sse({"done": True, "mock": True, "model": model})
            return

        params = self._with_messages(input_data, params)
        started = time.time()
        trace: list = []
        finish = "stop"
        try:
            async for kind, payload in self.run_tools(
                    list(params["messages"]), self._tools_on(params)):
                if kind == "delta":
                    yield _sse({"delta": payload})
                elif kind == "tool_start":
                    yield _sse({"tool": {
                        "id": payload["id"], "name": payload["name"],
                        "phase": "start", "args": payload["args"],
                    }})
                elif kind == "tool_end":
                    call, res, ms = payload
                    yield _sse({"tool": {
                        "id": call["id"], "phase": "end",
                        "ok": res["ok"], "detail": res["detail"], "ms": ms,
                    }})
                elif kind == "done":
                    trace = payload["trace"]
                    finish = payload["finish_reason"]
        except Exception as e:
            logger.error(f"NLP 流式推理失败: {e}")
            yield _sse({"error": f"推理失败: {e}"})
            return

        done = {
            "done": True,
            "model": model,
            "latency_ms": int((time.time() - started) * 1000),
            # 让前端能区分「答完了」和「被长度上限截断」—— 截断是用户唯一能
            # 察觉到的失败，不说明的话只会以为模型坏了
            "finish_reason": finish,
        }
        if trace:
            done["tool_calls"] = trace
        yield _sse(done)

    # ── 入口 2：OpenAI 格式（Cherry Studio 等外部客户端接入）──

    async def chat(self, messages: list, model: str = "qwen3",
                   max_tokens: int | None = None,
                   temperature: float | None = None,
                   tools_on: bool = True) -> dict:
        """非流式对话，返回 OpenAI chat.completion 结构"""
        model = self._resolve_model(model) or SUPPORTED_MODELS[0]
        started = time.time()
        messages = _ensure_system([dict(m) for m in messages])

        if self.should_mock():
            text, trace, usage, finish = mock_nlp(model)["output"], [], None, "stop"
        else:
            text, trace, usage, finish = "", [], None, "stop"
            async for kind, payload in self.run_tools(
                    messages, tools_on and self.tools_enabled,
                    max_tokens, temperature):
                if kind == "done":
                    text = payload["content"]
                    trace = payload["trace"]
                    usage = payload["usage"]
                    finish = payload["finish_reason"]

        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(started),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                # 透传真实原因：被 max_tokens 截断时是 length，客户端据此提示用户
                "finish_reason": finish,
            }],
            "usage": usage or {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            },
            # 非标准字段：外部客户端会忽略，网页/调试时能看出用了哪些工具
            "tool_calls_trace": trace,
        }

    async def chat_stream(self, messages: list, model: str = "qwen3",
                          max_tokens: int | None = None,
                          temperature: float | None = None,
                          tools_on: bool = True):
        """流式对话，产出 OpenAI SSE 帧（含结尾的 data: [DONE]）

        工具调用的过程**不转发给客户端** —— 工具在服务端执行完，客户端只看到
        最终的 content 流。这样任何 OpenAI 客户端都能直接用上联网/计算能力，
        不需要它自己实现 function calling。
        """
        model = self._resolve_model(model) or SUPPORTED_MODELS[0]
        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        messages = _ensure_system([dict(m) for m in messages])
        usage = None
        finish = "stop"

        def frame(delta: dict, finish=None) -> str:
            return "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }, ensure_ascii=False) + "\n\n"

        # 首帧先发 role：OpenAI 客户端普遍据此建立消息气泡
        yield frame({"role": "assistant", "content": ""})

        if self.should_mock():
            for ch in mock_nlp(model)["output"]:
                yield frame({"content": ch})
                await asyncio.sleep(0.02)
        else:
            async for kind, payload in self.run_tools(
                    messages, tools_on and self.tools_enabled,
                    max_tokens, temperature):
                if kind == "delta":
                    yield frame({"content": payload})
                elif kind == "done":
                    usage = payload["usage"]
                    finish = payload["finish_reason"]

        yield frame({}, finish=finish)
        # 用量帧：choices 是空数组，这是 OpenAI 的规范形状
        if usage:
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model, "choices": [], "usage": usage,
            }, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"
