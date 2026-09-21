"""Mock 输出生成器 — 为每类模型提供模拟响应

文本类模拟输出统一带 MOCK_TAG 前缀：当只有部分服务部署时（例如只起了 nlp 没起 mtran），
降级返回的模拟结果很容易被误认为真实翻译/识别结果。
"""

import random

# 统一前缀，任何渠道（前端 / curl / API）都能一眼识别
MOCK_TAG = "[模拟] "

MOCK_ASR_TEXTS = [
    "你好，这是一段语音识别的输出。",
    "今天天气不错，适合出去走走。",
    "本地小模型正在运行中。",
]

MOCK_TRANSLATE_TEXTS = {
    "en-zh": "这是一段翻译后的中文文本。",
    "zh-en": "This is a translated English text.",
}

# 静音 WAV（前端另有"模拟：生成一段静音"提示）
MOCK_TTS_AUDIO_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="

MOCK_NLP_RESPONSES = {
    "chat": "这是一个本地语言模型的回复，实际运行时将调用 SmolLM2 或 Qwen2.5 模型。",
    "summarize": "摘要：本文介绍了本地部署小模型的方案。",
    "classify": "分类结果：技术文档（置信度 0.95）",
}


def mock_asr(model: str = "whisper") -> dict:
    return {
        "output": MOCK_TAG + random.choice(MOCK_ASR_TEXTS),
        "model": model,
        "mock": True,
        "latency_ms": random.randint(80, 200),
    }


def mock_translate(model: str = "mtran", source: str = "en", target: str = "zh") -> dict:
    key = f"{source}-{target}"
    text = MOCK_TRANSLATE_TEXTS.get(key, f"[Mock translation {source}->{target}]")
    return {
        "output": MOCK_TAG + text,
        "model": model,
        "mock": True,
        "latency_ms": random.randint(50, 150),
    }


def mock_tts(model: str = "piper") -> dict:
    return {
        "output": MOCK_TTS_AUDIO_B64,
        "model": model,
        "mock": True,
        "latency_ms": random.randint(100, 250),
        "format": "wav",
    }


def mock_nlp(model: str = "qwen", task: str = "chat") -> dict:
    return {
        "output": MOCK_TAG + MOCK_NLP_RESPONSES.get(task, MOCK_NLP_RESPONSES["chat"]),
        "model": model,
        "mock": True,
        "latency_ms": random.randint(200, 500),
    }
