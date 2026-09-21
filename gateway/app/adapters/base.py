"""适配器基类 + Mock Mixin"""

import time
from abc import ABC, abstractmethod
from enum import Enum

import httpx
import logging

logger = logging.getLogger(__name__)

# 服务可用性缓存时长（秒）。失败态缓存更短，
# 这样后启动的服务（如 mtran 容器）就绪后能自动恢复，无需重启 gateway。
OK_TTL = 30.0
FAIL_TTL = 10.0


def probe_health(url: str, timeout: float = 2.0) -> bool:
    """探测服务 /health 是否返回 200"""
    try:
        return httpx.get(f"{url.rstrip('/')}/health", timeout=timeout).status_code == 200
    except Exception:
        return False


class MockMode(str, Enum):
    AUTO = "auto"
    ON = "1"
    OFF = "0"


class BaseServiceAdapter(ABC):
    """所有模型适配器的基类"""

    def __init__(self, service_url: str, service_name: str, mock_mode: str = "auto"):
        self.service_url = service_url.rstrip("/")
        self.service_name = service_name
        self.mock_mode = mock_mode
        self._available: bool | None = None
        self._checked_at: float = 0.0

    def should_mock(self) -> bool:
        """判断是否应使用 Mock 输出"""
        if self.mock_mode == MockMode.ON:
            return True
        if self.mock_mode == MockMode.OFF:
            return False
        # AUTO: 检查服务是否可用
        return not self.any_backend_available()

    def any_backend_available(self) -> bool:
        """是否有任一后端可用；多后端适配器（如 translate）可覆写"""
        return self.is_available()

    def is_available(self) -> bool:
        """检查真实模型服务是否可达（带 TTL 缓存，避免结果被永久固化）"""
        now = time.time()
        if self._available is not None:
            ttl = OK_TTL if self._available else FAIL_TTL
            if now - self._checked_at < ttl:
                return self._available
        self._available = probe_health(self.service_url)
        if not self._available:
            logger.info(f"[{self.service_name}] 服务不可达，降级为 Mock")
        self._checked_at = now
        return self._available

    async def call_service(self, payload: dict) -> dict:
        """调用真实模型服务"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.service_url}/infer",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            # error 响应没有「真实/模拟」之分，标 mock:false 会造成语义矛盾
            if "error" not in result:
                result.setdefault("mock", False)
            return result

    async def stream_service(self, payload: dict):
        """流式调用真实模型服务，逐帧转发 SSE

        与 call_service 的区别是全程不缓冲：用 client.stream 边收边发，
        等整个响应体到齐再返回就失去流式的意义了。
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.service_url}/infer/stream", json=payload
            ) as resp:
                resp.raise_for_status()
                # aiter_lines 按行切分，SSE 帧的空行会变成空字符串；这里重新组装成帧
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line + "\n\n"

    @abstractmethod
    async def infer(self, input_data: str, model: str = "", params: dict = None) -> dict:
        """执行推理，子类必须实现"""
        ...
