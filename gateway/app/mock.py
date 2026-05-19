"""Mock 输出生成器 — 为每类模型提供模拟响应"""

import random

MOCK_ASR_TEXTS = [
    "你好，这是一段语音识别的模拟输出。",
    "今天天气不错，适合出去走走。",
    "本地小模型正在运行中。",
]

MOCK_TRANSLATE_TEXTS = {
    "en-zh": "这是一段翻译后的中文文本。",
    "zh-en": "This is a translated English text.",
}

MOCK_OCR_TEXTS = [
    "检测到文本内容：这是一段 OCR 模拟识别结果。",
    "文档标题：本地 AI 工具包\n正文内容：支持多种 OCR 模型。",
]

MOCK_TTS_AUDIO_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="

MOCK_NLP_RESPONSES = {
    "chat": "这是一个本地语言模型的模拟回复。实际运行时将调用 SmolLM2 或 Qwen2.5 模型。",
    "summarize": "摘要：本文介绍了本地部署小模型的方案。",
    "classify": "分类结果：技术文档（置信度 0.95）",
}


def mock_asr(model: str = "whisper") -> dict:
    return {
        "output": random.choice(MOCK_ASR_TEXTS),
        "model": model,
        "mock": True,
        "latency_ms": random.randint(80, 200),
    }


def mock_translate(model: str = "mtran", source: str = "en", target: str = "zh") -> dict:
    key = f"{source}-{target}"
    return {
        "output": MOCK_TRANSLATE_TEXTS.get(key, f"[Mock translation {source}->{target}]"),
        "model": model,
        "mock": True,
        "latency_ms": random.randint(50, 150),
    }


def mock_ocr(model: str = "ppocr") -> dict:
    return {
        "output": random.choice(MOCK_OCR_TEXTS),
        "model": model,
        "mock": True,
        "latency_ms": random.randint(100, 300),
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
        "output": MOCK_NLP_RESPONSES.get(task, MOCK_NLP_RESPONSES["chat"]),
        "model": model,
        "mock": True,
        "latency_ms": random.randint(200, 500),
    }
