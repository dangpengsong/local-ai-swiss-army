#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "🎭 Local AI Swiss Army — 演示脚本"
echo "目标: $BASE_URL"
echo ""

PAYLOAD=$(mktemp)
trap 'rm -f "$PAYLOAD"' EXIT

# ── 测试音频 ──
# 不能用 Mock 时代的占位串（如 "mock_audio_base64"）：它不是合法 base64，真实服务
# 解码时会直接报 "Incorrect padding"，让人误以为部署失败。
# 这里优先用 TTS 合成一段真实语音 —— ASR 能识别出内容，顺带验证 TTS→ASR 闭环；
# TTS 不可用时回退为合成正弦波，它不含语音，ASR 返回空文本属预期。

prepare_audio() {
  local text="今天天气很好，适合出门散步。"
  local b64
  b64=$(curl -s -X POST "$BASE_URL/tts" -H "Content-Type: application/json" \
        -d "{\"input\": \"$text\", \"model\": \"piper\"}" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("output",""))' 2>/dev/null || true)
  if [ -n "$b64" ]; then
    TEST_AUDIO="$b64"
    AUDIO_KIND="TTS 合成语音「$text」"
    return 0
  fi
  TEST_AUDIO=$(python3 - <<'PYEOF'
import base64, io, math, struct
sr, dur, freq = 16000, 1.5, 440.0
n = int(sr * dur)
frames = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr))) for i in range(n))
buf = io.BytesIO()
buf.write(b"RIFF"); buf.write(struct.pack("<I", 36 + len(frames))); buf.write(b"WAVEfmt ")
buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
buf.write(b"data"); buf.write(struct.pack("<I", len(frames))); buf.write(frames)
print(base64.b64encode(buf.getvalue()).decode())
PYEOF
)
  AUDIO_KIND="合成正弦波（非语音 —— ASR 返回空文本是正常的，不代表部署有问题）"
}

# 把 base64 音频包成 JSON 再发请求。
# 走临时文件而非命令行参数：几秒语音的 base64 可达数十万字符，塞进 argv 有顶到 ARG_MAX 的风险。
post_audio() {
  local path=$1 model=${2:-}
  printf '%s' "$TEST_AUDIO" | python3 -c '
import json, sys
payload = {"input": sys.stdin.read()}
if sys.argv[1]:
    payload["model"] = sys.argv[1]
print(json.dumps(payload))
' "$model" > "$PAYLOAD"
  curl -s -X POST "$BASE_URL$path" -H "Content-Type: application/json" --data-binary "@$PAYLOAD"
}

echo "0️⃣ 准备测试音频（第 3、7 步使用）..."
prepare_audio
echo "   → $AUDIO_KIND"
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
post_audio "/asr" whisper | python3 -m json.tool

# 翻译
echo ""
echo "4️⃣ 翻译测试..."
curl -s -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "mtran", "params": {"source": "en", "target": "zh"}}' | python3 -m json.tool

# TTS
echo ""
echo "5️⃣ TTS 测试..."
curl -s -X POST "$BASE_URL/tts" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "model": "piper"}' | python3 -m json.tool

# NLP（用推荐主力 Qwen3-4B；纯 CPU 约 7 tok/s，本步需等十几秒属正常）
echo ""
echo "6️⃣ NLP 测试..."
curl -s -X POST "$BASE_URL/nlp" \
  -H "Content-Type: application/json" \
  -d '{"input": "介绍一下你自己", "model": "qwen3", "params": {"task": "chat"}}' | python3 -m json.tool

# 语音管线
echo ""
echo "7️⃣ 语音管线测试...（ASR → NLP → TTS，输出含 base64 音频，较长）"
post_audio "/pipeline/voice" | python3 -m json.tool

# 自定义管线
echo ""
echo "8️⃣ 自定义管线测试..."
curl -s -X POST "$BASE_URL/pipeline/custom" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "steps": ["translate:mtran", "tts:piper"]}' | python3 -m json.tool

echo ""
echo "✅ 演示完成！"
