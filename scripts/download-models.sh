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

mkdir -p models/{asr,tts,nlp,mtran}

download() {
    local name=$1 url=$2 output=$3
    if [ -f "$output" ] && [ "$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)" -gt 0 ]; then
        echo "  ⏭ $name 已存在: $output"
        return 0
    fi
    echo "  📥 下载 $name ..."
    # 替换 huggingface.co 为镜像
    local mirror_url="${url/https:\/\/huggingface.co/$HF_MIRROR}"
    # 先写 .part，下完再改名。gateway 的就绪判定只看「文件存在且非空」，直接写目标文件
    # 会让下到一半的模型在面板上显示绿色「已就绪」，点推理才报错。上轮中断的残留也一并清掉。
    rm -f "$output.part"
    wget --no-check-certificate -q --show-progress -O "$output.part" "$mirror_url" 2>&1 || {
        echo "  ❌ 下载失败，尝试 curl ..."
        curl -L -k -o "$output.part" "$mirror_url" 2>&1 || {
            echo "  ❌ curl 也失败了，请手动下载"
            rm -f "$output.part"
            return 1
        }
    }
    mv "$output.part" "$output"
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
    echo "  ℹ️  Argos: 语言包已内置在镜像里（构建时下载），无需在此处理"
    echo "  ⚠️  MTranServer: 使用独立 Docker 容器；翻译模型在首次翻译时自动下载到 models/mtran/"
    echo ""
}

download_tts() {
    echo "── TTS 模型 ──"
    local piper_base="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN"
    mkdir -p models/tts/piper models/tts/g2pW

    # 默认语音：小雅（拼音方案，停顿更自然）
    download "Piper 中文语音 zh_CN-xiao_ya-medium (.onnx)" \
        "$piper_base/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx" \
        "models/tts/piper/zh_CN-xiao_ya-medium.onnx"
    download "Piper 中文语音 zh_CN-xiao_ya-medium (.onnx.json)" \
        "$piper_base/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx.json" \
        "models/tts/piper/zh_CN-xiao_ya-medium.onnx.json"

    # 备选语音：华言（espeak 方案，中英混读略好）
    download "Piper 中文语音 zh_CN-huayan-medium (.onnx)" \
        "$piper_base/huayan/medium/zh_CN-huayan-medium.onnx" \
        "models/tts/piper/zh_CN-huayan-medium.onnx"
    download "Piper 中文语音 zh_CN-huayan-medium (.onnx.json)" \
        "$piper_base/huayan/medium/zh_CN-huayan-medium.onnx.json" \
        "models/tts/piper/zh_CN-huayan-medium.onnx.json"

    # 小雅依赖的 g2pW 注音模型（约 152MB，仅拼音方案需要）
    download "g2pW 中文注音模型（约 152MB）" \
        "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/zh/zh_CN/_resources/g2pw.tar.gz" \
        "models/tts/g2pW/g2pw.tar.gz"
    if [ -f models/tts/g2pW/g2pw.tar.gz ] && [ ! -f models/tts/g2pW/g2pw.onnx ]; then
        echo "  📦 解压 g2pW 注音模型 ..."
        tar xzf models/tts/g2pW/g2pw.tar.gz -C models/tts/g2pW && rm -f models/tts/g2pW/g2pw.tar.gz
        echo "  ✅ g2pW 解压完成"
    fi
    echo "  ⚠️  OuteTTS / OpenAudio: 需手动下载"
    echo ""
}

download_nlp() {
    echo "── NLP 模型 ──"
    download "Qwen3-4B-Instruct-2507-GGUF（约 2.3GB，推荐主力）" \
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
        "models/nlp/qwen3-4b-instruct-2507-q4_k_m.gguf"
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
    tts)       download_tts ;;
    nlp)       download_nlp ;;
    all)
        download_asr
        download_translate
        download_tts
        download_nlp
        ;;
    *)
        echo "用法: $0 [asr|translate|tts|nlp|all]"
        exit 1
        ;;
esac

echo "============================================"
echo "  ✅ 下载完成"
echo "============================================"
echo ""
echo "启动服务："
echo "  docker compose --profile full up -d --build"
