#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "🎭 Local AI Swiss Army — 演示脚本"
echo "目标: $BASE_URL"
echo ""

# 健康检查
echo "1️⃣ 健康检查..."
curl -s "$BASE_URL/health" | python3 -m json.tool

# 服务状态
echo ""
echo "2️⃣ 服务状态..."
curl -s "$BASE_URL/status" | python3 -m json.tool

# ASR
echo ""
echo "3️⃣ ASR 测试..."
curl -s -X POST "$BASE_URL/asr" \
  -H "Content-Type: application/json" \
  -d '{"input": "mock_audio_base64", "model": "whisper"}' | python3 -m json.tool

# 翻译
echo ""
echo "4️⃣ 翻译测试..."
curl -s -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "mtran", "params": {"source": "en", "target": "zh"}}' | python3 -m json.tool

# OCR
echo ""
echo "5️⃣ OCR 测试..."
curl -s -X POST "$BASE_URL/ocr" \
  -H "Content-Type: application/json" \
  -d '{"input": "mock_image_base64", "model": "ppocr"}' | python3 -m json.tool

# TTS
echo ""
echo "6️⃣ TTS 测试..."
curl -s -X POST "$BASE_URL/tts" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "model": "piper"}' | python3 -m json.tool

# NLP
echo ""
echo "7️⃣ NLP 测试..."
curl -s -X POST "$BASE_URL/nlp" \
  -H "Content-Type: application/json" \
  -d '{"input": "介绍一下你自己", "model": "qwen", "params": {"task": "chat"}}' | python3 -m json.tool

# 语音管线
echo ""
echo "8️⃣ 语音管线测试..."
curl -s -X POST "$BASE_URL/pipeline/voice" \
  -H "Content-Type: application/json" \
  -d '{"input": "mock_audio_base64"}' | python3 -m json.tool

# 文档管线
echo ""
echo "9️⃣ 文档管线测试..."
curl -s -X POST "$BASE_URL/pipeline/document" \
  -H "Content-Type: application/json" \
  -d '{"input": "mock_image_base64"}' | python3 -m json.tool

# 自定义管线
echo ""
echo "🔟 自定义管线测试..."
curl -s -X POST "$BASE_URL/pipeline/custom" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "steps": ["translate:mtran", "tts:piper"]}' | python3 -m json.tool

echo ""
echo "✅ 演示完成！"
