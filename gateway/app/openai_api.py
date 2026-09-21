"""OpenAI 兼容接口 — 供 Cherry Studio 等外部客户端接入

llama.cpp server 自己就暴露 /v1/chat/completions，但**不能**把客户端请求直接
透传过去，有三个理由：

1. **工具调用**。四个工具（联网搜索、当前时间、计算器、网页抓取）的循环跑在
   gateway 里。客户端若自己实现 function calling，就得先收 tool_calls、执行、
   再回填 —— 而 Cherry Studio 这类客户端是当聊天用的，不会做这件事。所以工具
   在**服务端执行完**，客户端只看到最终的 content 流，配好就能用。
2. **模型名**。llama.cpp 返回的 model 字段是文件路径
   （/models/qwen3-4b-instruct-2507-q4_k_m.gguf），客户端要的是 qwen3。
   不映射的话 Cherry Studio 会因为「返回的模型和请求的不一致」报错。
3. **system 提示**。客户端不带 system 时补一个默认的，否则 Qwen3 裸跑会漂。

所以这里是一个适配层，不是反向代理。适配层的活干在 NLPAdapter.chat /
chat_stream 里，本文件只负责 HTTP 形状：路由、请求校验、错误格式。

设计取舍：**忽略客户端自带的 tools 定义**。那些定义意味着「客户端想自己执行
工具」，而我们已经在服务端执行了自己的工具；两套混在一起会出现客户端收到
tool_calls 却按自己的定义去执行的错配。客户端带不带 tools，行为都一样。
"""

import logging
import time

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .adapters.nlp import MAX_TOKENS_CAP, SUPPORTED_MODELS
from .config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])

# /v1/models 里 created 字段用的时间戳，取进程启动时刻
_STARTED = int(time.time())

# 上下文长度，非标准字段但被不少客户端读取用于显示/截断
CONTEXT_LENGTH = 4096


class ChatMessage(BaseModel):
    role: str
    # content 可以是 null（带 tool_calls 的 assistant 轮）或多模态数组，
    # 统一在 _content_to_text 里收敛成字符串
    content: object = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI 请求体里我们真正用到的字段子集

    其余字段（top_p / presence_penalty / stop / stream_options ...）由 pydantic
    默认忽略 —— 客户端爱发什么发什么，不认识的直接丢掉，不报错。
    """
    model: str = ""
    messages: list[ChatMessage] = []
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # 接住但不用，见模块开头的说明
    tools: list | None = None
    tool_choice: object = None


def _content_to_text(content) -> str:
    """把任意形状的 content 收敛成纯文本

    多模态数组（[{"type": "text", "text": "..."}]）只取文字部分；图片等其余
    类型直接丢弃 —— 本地跑的是纯文本模型，喂不进去。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        return "".join(parts)
    return str(content)


def _error(message: str, status: int, code: str = "invalid_request_error"):
    """OpenAI 形状的错误响应

    必须是 {"error": {...}} 而不是 FastAPI 默认的 {"detail": ...}：客户端普遍
    按前者解析，后者只会被显示成「未知错误」，用户看不到真正的原因。
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _check_auth(authorization: str | None):
    """可选的 API Key 校验

    默认不配 key 就不校验 —— 本地私有部署，gateway 的其他接口本来也没有鉴权，
    单独给这一个加锁没有意义。要暴露到局域网外时在 .env 填 OPENAI_API_KEY。
    """
    expected = get_settings().openai_api_key
    if not expected:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _nlp():
    """取 NLP 适配器实例

    延迟导入：adapters 定义在 main.py，而 main.py 会 include 本模块的 router，
    模块级导入会成环。放在函数里，等 main 加载完毕后再取就没事。
    """
    from .main import adapters
    return adapters["nlp"]


def _sampling(req: ChatCompletionRequest) -> tuple[int | None, float | None]:
    """收敛客户端给的采样参数

    max_tokens 必须封顶：上下文只有 4096，客户端默认可能填 4096 甚至更大，
    放开会把上下文打爆（llama.cpp 会直接截断 prompt，表现成「答非所问」）。
    temperature 夹到 [0, 2]，负数和 100 都是无效值。
    """
    max_tokens = None
    if req.max_tokens and req.max_tokens > 0:
        max_tokens = min(req.max_tokens, MAX_TOKENS_CAP)

    temperature = None
    if req.temperature is not None:
        temperature = min(max(req.temperature, 0.0), 2.0)

    return max_tokens, temperature


@router.get("/v1/models")
async def list_models(authorization: str | None = Header(None)):
    """模型列表。Cherry Studio 的「检查连接」按钮打的就是这个接口。"""
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": _STARTED,
                "owned_by": "local-ai-swiss-army",
                "context_length": CONTEXT_LENGTH,
            }
            for model_id in SUPPORTED_MODELS
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(None),
):
    """对话补全，支持 stream: true / false

    工具在服务端执行，客户端只看到 content —— 所以任何 OpenAI 客户端都能直接
    用上联网/计算能力，不需要它自己实现 function calling。
    """
    _check_auth(authorization)

    model = req.model or SUPPORTED_MODELS[0]
    if model not in SUPPORTED_MODELS:
        return _error(
            f"模型 {model!r} 不存在，可用模型: {', '.join(SUPPORTED_MODELS)}",
            404,
            "model_not_found",
        )
    if not req.messages:
        return _error("messages 不能为空", 400)

    messages = [
        {"role": m.role, "content": _content_to_text(m.content)}
        for m in req.messages
    ]
    max_tokens, temperature = _sampling(req)
    adapter = _nlp()

    if req.stream:
        return StreamingResponse(
            adapter.chat_stream(messages, model, max_tokens, temperature),
            media_type="text/event-stream",
            # 与 /nlp/stream 同理：防止将来挂 nginx 时把 SSE 缓冲成一次性输出
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await adapter.chat(messages, model, max_tokens, temperature)
    except Exception as e:
        # 不把异常直接抛给客户端：FastAPI 的 500 响应体不是 OpenAI 形状，
        # 客户端会显示成「未知错误」，用户看不到真正的原因
        logger.error(f"OpenAI 兼容接口推理失败: {e}")
        return _error(f"推理失败: {e}", 502, "upstream_error")
