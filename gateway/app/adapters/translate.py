"""翻译适配器 — MTranServer（独立 Docker 容器）

Argos Translate 已于 2026-09 移除：其 en→zh 训练语料混入字幕文件，会把 ASS 样式标签
（形如 ``{\\fn华文楷体\\fs16\\1cHE0E0E0}``）当正文吐出来，simple 词条也大量错译。
16 组样本对比中 MTranServer 每组都不劣于它，故删除而不是保留降级选项。
"""

import time

import httpx
import logging

from .base import BaseServiceAdapter, MockMode
from ..mock import mock_translate

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["mtran"]

# MTranServer 使用 zh-Hans/zh-Hant 而非 zh
LANG_MAP = {"zh": "zh-Hans", "zh-CN": "zh-Hans", "zh-TW": "zh-Hant"}


class TranslateAdapter(BaseServiceAdapter):
    """MTranServer 跑在独立容器里，接口是 /translate 而非基类约定的 /infer，
    所以 service_url 直接指向该容器 —— 探测（/health）与调用都走它，
    is_available / should_mock 无需覆写即可正确工作。"""

    def __init__(self, mtran_url: str, mock_mode: str = "auto"):
        super().__init__(mtran_url, "Translate", mock_mode)

    async def infer(self, input_data: str, model: str = "mtran", params: dict = None) -> dict:
        params = params or {}
        source = params.get("source", "en")
        target = params.get("target", "zh")

        if model not in SUPPORTED_MODELS:
            return {"error": f"不支持的模型: {model}，可选: {SUPPORTED_MODELS}"}

        if self.mock_mode == MockMode.ON:
            return mock_translate(model, source, target)

        if self.mock_mode == MockMode.AUTO and not self.is_available():
            return mock_translate(model, source, target)

        return await self._call_mtran(input_data, source, target)

    async def _call_mtran(self, text: str, source: str, target: str) -> dict:
        """直接调用 MTranServer 容器

        超时给到 300 秒：某个语言对首次翻译时，服务端要先把该语言对的模型下载下来。
        """
        start = time.time()
        src = LANG_MAP.get(source, source)
        tgt = LANG_MAP.get(target, target)
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.service_url}/translate",
                    json={"from": src, "to": tgt, "text": text, "html": False},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "output": data.get("result", ""),
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
