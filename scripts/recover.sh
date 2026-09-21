#!/usr/bin/env bash
# 重启 Docker Desktop（或 wsl --shutdown）后，容器的 bind mount 可能失效：
# 宿主机目录里明明有文件，容器内却是空目录，而 docker inspect 显示的挂载配置
# 完全正确 —— 所以只能从容器内部看实际内容，才判断得出来。
#
# 症状通常是 Web 面板报「Web 面板未找到」，或 ASR／TTS／翻译报找不到模型。
#
# 这个脚本逐个容器检查挂载点有没有内容，空的就重建那个容器。
# 幂等：正常的容器一律不碰，可以随时重复执行。
#
#   bash scripts/recover.sh
#
# 注意：重建用的是 --force-recreate，只删容器重建容器 —— 镜像不重建、不拉取，
# 模型也不会重新下载，放心跑。
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

# Docker 没在跑就先退出：这时容器是「不存在」而不是「挂载坏了」，照常往下走
# 会去创建容器，把「Docker Desktop 还没起来」误报成故障。
if ! docker info >/dev/null 2>&1; then
  echo "Docker 未运行，无需恢复"
  exit 0
fi

# 容器名|compose 服务名|要检查的挂载点（可有多个）
ENTRIES=(
  "la-gateway|gateway|/app/web|/app/models"
  "la-tts|tts|/models"
  "la-asr|asr|/models"
  "la-nlp|nlp|/models"
  "la-mtran|mtran|/models"
)

# 重建后要等容器真正跑起来再复检；刚 up 完就 docker exec 会扑空
wait_running() {
  for _ in $(seq 1 15); do
    docker exec "$1" true 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

# 挂载点里有内容就算正常。用 ls -A 是因为 TTS 的 /models/.hf 这类隐藏目录
# 也是有效内容，用 ls 会漏判。
mount_ok() {
  [ -n "$(docker exec "$1" ls -A "$2" 2>/dev/null)" ]
}

fixed=0
failed=0

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r -a parts <<< "$entry"
  ctn="${parts[0]}"; svc="${parts[1]}"; mounts=("${parts[@]:2}")

  if [ "$(docker inspect -f '{{.State.Running}}' "$ctn" 2>/dev/null)" = "true" ]; then
    bad=""
    for m in "${mounts[@]}"; do
      mount_ok "$ctn" "$m" || { bad="$m"; break; }
    done
    if [ -z "$bad" ]; then
      echo "  ✅ $ctn  正常"
      continue
    fi
    echo "  🔧 $ctn  $bad 为空 → 重建容器"
    docker compose --profile full up -d --force-recreate "$svc" >/dev/null 2>&1
  else
    echo "  ▶️  $ctn  未运行 → 启动"
    docker compose --profile full up -d "$svc" >/dev/null 2>&1
  fi

  # 复检：重建后仍为空，说明宿主机那边的源目录本身就没文件，不是挂载的锅。
  # 得说清楚，否则一句「已恢复」会把人带偏。
  if wait_running "$ctn"; then
    still=""
    for m in "${mounts[@]}"; do
      mount_ok "$ctn" "$m" || { still="$m"; break; }
    done
    if [ -z "$still" ]; then
      fixed=$((fixed + 1))
    else
      echo "  ⚠️  $ctn 的 $still 仍为空 —— 检查宿主机上对应的源目录是不是真的没文件"
      failed=$((failed + 1))
    fi
  else
    echo "  ⚠️  $ctn 起不来，看 docker logs $ctn"
    failed=$((failed + 1))
  fi
done

echo ""
if [ "$fixed" -eq 0 ] && [ "$failed" -eq 0 ]; then
  echo "🎉 全部正常，未做任何改动"
elif [ "$failed" -eq 0 ]; then
  echo "✅ 已恢复 $fixed 个容器"
else
  echo "⚠️  恢复了 $fixed 个，$failed 个仍有问题"
fi
