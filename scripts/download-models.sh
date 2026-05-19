#!/usr/bin/env bash
set -euo pipefail

# HF 镜像（国内可用）
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

CATEGORY="${1:-all}"

echo "============================================"
echo "  模型权重下载工具"
echo "  镜像源: $HF_MIRROR"
echo "  类别: $CATEGORY"
echo "============================================"
echo ""

mkdir -p models/{asr,translate,ocr,tts,nlp}

download() {
    local name=$1 url=$2 output=$3
    if [ -f "$output" ] && [ "$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)" -gt 0 ]; then
        echo "  ⏭ $name 已存在: $output"
        return 0
    fi
    echo "  📥 下载 $name ..."
    # 替换 huggingface.co 为镜像
    local mirror_url="${url/https:\/\/huggingface.co/$HF_MIRROR}"
    wget --no-check-certificate -q --show-progress -O "$output" "$mirror_url" 2>&1 || {
        echo "  ❌ 下载失败，尝试 curl ..."
        curl -L -k -o "$output" "$mirror_url" 2>&1 || {
            echo "  ❌ curl 也失败了，请手动下载"
            rm -f "$output"
            return 1
        }
    }
    echo "  ✅ $name 下载完成"
}

download_asr() {
    echo "── 语音识别模型 ──"
    mkdir -p models/asr/whisper-base-ct2
    download "whisper-base-ct2-model" \
        "https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin" \
        "models/asr/whisper-base-ct2/model.bin"
    download "whisper-base-ct2-config" \
        "https://huggingface.co/Systran/faster-whisper-base/resolve/main/config.json" \
        "models/asr/whisper-base-ct2/config.json"
    download "whisper-base-ct2-tokenizer" \
        "https://huggingface.co/Systran/faster-whisper-base/resolve/main/tokenizer.json" \
        "models/asr/whisper-base-ct2/tokenizer.json"
    download "whisper-base-ct2-vocabulary" \
        "https://huggingface.co/Systran/faster-whisper-base/resolve/main/vocabulary.txt" \
        "models/asr/whisper-base-ct2/vocabulary.txt"
    echo ""
}

download_translate() {
    echo "── 翻译模型 ──"
    echo "  ⚠️  MTranServer: 使用独立 Docker 容器，无需下载"
    echo "  ⚠️  Argos Translate: 首次运行时自动安装语言包"
    echo ""
}

download_ocr() {
    echo "── OCR 模型 ──"
    echo "  ⚠️  Tesseract: 语言包在 Docker 构建时安装"
    echo "  ⚠️  PPOCR: 首次运行时自动下载模型"
    echo ""
}

download_tts() {
    echo "── TTS 模型 ──"
    echo "  ⚠️  Piper: 需手动下载 .onnx 模型到 models/tts/"
    echo "  ⚠️  OuteTTS / OpenAudio: 需手动下载"
    echo ""
}

download_nlp() {
    echo "── NLP 模型 ──"
    download "Qwen2.5-0.5B-GGUF" \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
        "models/nlp/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    download "SmolLM2-1.7B-GGUF" \
        "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf" \
        "models/nlp/smollm2.gguf"
    echo ""
}

case "$CATEGORY" in
    asr)       download_asr ;;
    translate) download_translate ;;
    ocr)       download_ocr ;;
    tts)       download_tts ;;
    nlp)       download_nlp ;;
    all)
        download_asr
        download_translate
        download_ocr
        download_tts
        download_nlp
        ;;
    *)
        echo "用法: $0 [asr|translate|ocr|tts|nlp|all]"
        exit 1
        ;;
esac

echo "============================================"
echo "  ✅ 下载完成"
echo "============================================"
echo ""
echo "启动服务："
echo "  docker compose --profile full up -d --build"
