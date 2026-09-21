"""构建期把 argos 语言包装进镜像（只由 Dockerfile 调用，不参与运行时）

为什么不用 argos 自带的 get_available_packages() 挑包：那条路要先拉索引服务，
索引本身可能不可达；而且它默认取最新版，上游发新版会让不同时间构建出的镜像
内容不一致。这里写死带版本号的直链，构建出来的镜像可追溯。

为什么要自己写重试：单个语言包 70–115MB，argos-net.com 在大文件传输中途会
断流（IncompleteRead）。重试 10 次仍失败则让构建直接失败 —— 不要静默跳过，
否则镜像看着构建成功、实际没有语言包，要等运行时翻译才报错。
"""

import os
import shutil
import sys
import time
from pathlib import Path

import requests
from argostranslate.package import install_from_path

BASE_URL = "https://argos-net.com/v1/translate-{}.argosmodel"

# 文件名里的版本号与 scripts/download-models.sh 曾经使用的一致
PACKAGES = [
    ("en_zh-1_9", "英 → 中"),
    ("zh_en-1_9", "中 → 英"),
    ("en_ja-1_1", "英 → 日"),
    ("ja_en-1_1", "日 → 英"),
]

TMP_DIR = Path("/tmp/argos-download")
RETRIES = 10
CHUNK = 1 << 20  # 1MB


def download(name: str) -> Path:
    """下载语言包，断流则重试；返回本地文件路径"""
    url = BASE_URL.format(name)
    dest = TMP_DIR / f"{name}.argosmodel"

    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        f.write(chunk)
            return dest
        except Exception as e:
            if attempt == RETRIES:
                raise
            print(f"    第 {attempt}/{RETRIES} 次失败：{e}", flush=True)
            time.sleep(2)

    raise AssertionError("unreachable")


def main() -> int:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name, label in PACKAGES:
            print(f"  安装 argos 语言包 {label}（{name}）...", flush=True)
            path = download(name)
            install_from_path(str(path))
            path.unlink()
        print("  ✅ argos 语言包全部安装完成", flush=True)
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
