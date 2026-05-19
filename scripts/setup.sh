#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  本地AI工具箱 — 一键安装"
echo "  来源：公众号「名侦探科男」"
echo "============================================"
echo ""

# 检查 Docker
if ! command -v docker &>/dev/null; then
    echo "❌ 未检测到 Docker，请先安装 Docker Desktop"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "❌ 未检测到 Docker Compose V2"
    exit 1
fi

echo "✅ Docker 环境检查通过"
echo ""

# 1. 创建目录
echo "📁 创建目录..."
mkdir -p models/{asr,translate,ocr,tts,nlp}

# 2. 环境变量
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ 已创建 .env"
else
    echo "⏭ .env 已存在"
fi

# 3. 构建 Gateway（始终需要）
echo ""
echo "🔨 构建 Gateway..."
docker compose build gateway

echo ""
echo "============================================"
echo "  ✅ 安装完成！"
echo "============================================"
echo ""
echo "启动方式："
echo ""
echo "  【Mock 模式】无需模型，秒级启动："
echo "    docker compose up -d gateway"
echo "    → http://localhost:8000"
echo ""
echo "  【完整模式】下载模型后启动："
echo "    ./scripts/download-models.sh all"
echo "    docker compose --profile full up -d --build"
echo "    → http://localhost:8000"
echo ""
