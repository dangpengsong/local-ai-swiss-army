"""翻译模型服务 — Argos Translate（MTranServer 通过独立 Docker 容器提供）"""

import time
import logging

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_argos_installed = False
try:
    import argostranslate.translate
    _argos_installed = True
except ImportError:
    logger.info("argos-translate 未安装")


def _installed_pairs() -> list:
    """已安装的 argos 语言对，形如 ["en->zh", "zh->en"]

    用 get_installed_packages() 而不是遍历 get_installed_languages()：
    后者的 translations_from 元素是 CachedTranslation，属性名是 to_lang（Language
    对象），既没有 to_code 也没有 code —— 取错属性会抛 AttributeError 被下面的
    except 吞掉，结果语言包明明装了却报告未就绪。且该列表里混有
    IdentityTranslation（zh->zh 这类恒等项），并不是真实安装的语言包。
    """
    if not _argos_installed:
        return []
    try:
        from argostranslate.package import get_installed_packages
        return [f"{p.from_code}->{p.to_code}" for p in get_installed_packages()]
    except Exception as e:
        logger.warning(f"读取 argos 语言包失败: {e}")
        return []


app = FastAPI(title="Translate Service")


def _check_language_pair(source: str, target: str) -> bool:
    """检查 argos 语言对是否已安装"""
    return f"{source}->{target}" in _installed_pairs()


class InferRequest(BaseModel):
    input: str
    model: str = "argos"
    params: dict = None


@app.get("/health")
async def health():
    pairs = _installed_pairs()
    models = {
        # 装上 argos 包 ≠ 能用：语言包是构建期装进镜像的，万一没装进来，
        # 翻译必然失败，不能只因为 import 成功就报就绪（网关会据此把模型标成绿色）
        "argos": {
            "weight_ready": bool(pairs),
            "engine": "argos-translate",
            "pairs": pairs,
        },
        "mtran": {"weight_ready": False, "note": "独立 Docker 容器"},
    }
    return {"status": "ok", "models": models}


@app.post("/infer")
async def infer(req: InferRequest):
    start = time.time()
    params = req.params or {}
    source = params.get("source", "en")
    target = params.get("target", "zh")

    if req.model == "argos" and _argos_installed:
        # 先查语言包，比等 translate() 抛异常再猜原因更明确
        if not _check_language_pair(source, target):
            installed = _installed_pairs()
            return {
                "error": (
                    f"argos 未安装语言包 {source}->{target}"
                    f"（已安装: {', '.join(installed) if installed else '无'}）"
                ),
                "model": req.model,
                "latency_ms": int((time.time() - start) * 1000),
            }
        try:
            from argostranslate.translate import translate
            output = translate(req.input, source, target)
            return {
                "output": output,
                "model": "argos",
                "latency_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            logger.error(f"Argos 翻译失败: {e}")
            # 失败一律走 error 字段：塞进 output 会被标记为"真实"结果展示
            return {
                "error": f"Argos 翻译失败: {e}",
                "model": req.model,
                "latency_ms": int((time.time() - start) * 1000),
            }

    # MTranServer 走独立容器，这里只处理 argos
    return {
        "error": f"翻译模型 {req.model} 不可用：argos 服务仅处理 argos 模型，请确认模型名与语言包 {source}->{target}",
        "model": req.model,
        "latency_ms": int((time.time() - start) * 1000),
    }
