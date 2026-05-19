"""翻译适配器 — MTranServer / Argos Translate"""

import httpx
import logging

from .base import BaseServiceAdapter
from ..mock import mock_translate

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["mtran", "argos"]


class TranslateAdapter(BaseServiceAdapter):
    def __init__(self, service_url: str, mock_mode: str = "auto", mtran_url: str = ""):
        super().__init__(service_url, "Translate", mock_mode)
        self.mtran_url = mtran_url.rstrip("/") if mtran_url else ""

    async def infer(self, input_data: str, model: str = "mtran", params: dict = None) -> dict:
        params = params or {}
        source = params.get("source", "en")
        target = params.get("target", "zh")

        if model not in SUPPORTED_MODELS:
            return {"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"}

        if self.should_mock():
            return mock_translate(model, source, target)

        # mtran 走独立容器
        if model == "mtran" and self.mtran_url:
            return await self._call_mtran(input_data, source, target)

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
            return {
                "output": f"[MTranServer 不可用: {str(e)}]",
                "model": "mtran",
                "mock": False,
                "latency_ms": int((time.time() - start) * 1000),
            }
