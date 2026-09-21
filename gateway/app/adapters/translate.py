"""翻译适配器 — MTranServer / Argos Translate"""

import time

import httpx
import logging

from .base import BaseServiceAdapter, MockMode, probe_health, OK_TTL, FAIL_TTL
from ..mock import mock_translate

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["mtran", "argos"]


class TranslateAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto", mtran_url: str = ""):
        super().__init__(service_url, "Translate", mock_mode)
        self.mtran_url = mtran_url.rstrip("/") if mtran_url else ""
        self._mtran_available: bool | None = None
        self._mtran_checked_at: float = 0.0

    def mtran_available(self) -> bool:
        """探测 MTranServer 独立容器（与 argos 的 service_url 无关，带 TTL 缓存）"""
        now = time.time()
        if self._mtran_available is not None:
            ttl = OK_TTL if self._mtran_available else FAIL_TTL
            if now - self._mtran_checked_at < ttl:
                return self._mtran_available
        self._mtran_available = probe_health(self.mtran_url)
        if not self._mtran_available:
            logger.info("[Translate] MTranServer 不可达，降级为 Mock")
        self._mtran_checked_at = now
        return self._mtran_available

    def any_backend_available(self) -> bool:
        """argos 服务或 MTranServer 任一可用即算可用（供 /status 展示）"""
        return self.is_available() or (bool(self.mtran_url) and self.mtran_available())

    async def infer(self, input_data: str, model: str = "mtran", params: dict = None) -> dict:
        params = params or {}
        source = params.get("source", "en")
        target = params.get("target", "zh")

        if model not in SUPPORTED_MODELS:
            return {"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"}

        if self.mock_mode == MockMode.ON:
            return mock_translate(model, source, target)

        # mtran 走独立容器：可用性只看它自己，与 argos 服务（translate:8002）无关
        if model == "mtran" and self.mtran_url:
            if self.mock_mode == MockMode.AUTO and not self.mtran_available():
                return mock_translate(model, source, target)
            return await self._call_mtran(input_data, source, target)

        if self.mock_mode == MockMode.AUTO and not self.is_available():
            return mock_translate(model, source, target)

        payload = {
            "input": input_data,
            "model": model,
            "params": {"source": source, "target": target, **params},
        }
        return await self.call_service(payload)

    async def _call_mtran(self, text: str, source: str, target: str) -> dict:
        """直接调用 MTranServer 容器"""
        import time
        start = time.time()
        # MTranServer 使用 zh-Hans/zh-Hant 而非 zh
        lang_map = {"zh": "zh-Hans", "zh-CN": "zh-Hans", "zh-TW": "zh-Hant"}
        src = lang_map.get(source, source)
        tgt = lang_map.get(target, target)
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.mtran_url}/translate",
                    json={"from": src, "to": tgt, "text": text, "html": False},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                translated = data.get("result", "")
                return {
                    "output": translated,
                    "model": "mtran",
                    "mock": False,
                    "latency_ms": int((time.time() - start) * 1000),
                }
        except Exception as e:
            logger.error(f"MTranServer 调用失败: {e}")
            # 走 error 字段：带 output 返回会被前端当作"真实推理结果"显示绿色徽章
            return {
                "error": f"MTranServer 调用失败: {e}",
                "model": "mtran",
                "latency_ms": int((time.time() - start) * 1000),
            }
