"""适配器基类 + Mock Mixin"""

from abc import ABC, abstractmethod
from enum import Enum

import httpx
import logging

logger = logging.getLogger(__name__)


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

    def should_mock(self) -> bool:
        """判断是否应使用 Mock 输出"""
        if self.mock_mode == MockMode.ON:
            return True
        if self.mock_mode == MockMode.OFF:
            return False
        # AUTO: 检查服务是否可用
        return not self.is_available()

    def is_available(self) -> bool:
        """检查真实模型服务是否可达"""
        if self._available is not None:
            return self._available
        try:
            resp = httpx.get(f"{self.service_url}/health", timeout=2.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
            logger.info(f"[{self.service_name}] 服务不可达，降级为 Mock")
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
            result.setdefault("mock", False)
            return result

    @abstractmethod
    async def infer(self, input_data: str, model: str = "", params: dict = None) -> dict:
        """执行推理，子类必须实现"""
        ...
