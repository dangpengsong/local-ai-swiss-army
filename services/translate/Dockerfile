FROM python:3.12-slim

WORKDIR /app

# 国内网络默认走清华源；海外环境可改为 https://pypi.org/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# CPU 版 torch 只在 PyTorch 官方源提供。少了这个源，pip 会装 GPU 版 torch，
# 连带 2GB+ 的 nvidia-* CUDA 包（本项目纯 CPU 推理，用不上）。
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} --extra-index-url ${TORCH_CPU_INDEX_URL} -r requirements.txt

# argos 语言包（en↔zh、en↔ja，约 365MB）在构建期装进镜像，落在 argos 的默认目录
# /root/.local/share/argos-translate/packages 下 —— 容器一起来就能翻译，不依赖宿主机
# 挂载，也没有启动时解压那一步。
# 下载逻辑见 install_argos_packages.py：固定版本的直链 + 重试；下载不全就让构建失败，
# 不会留下一个「能启动但翻译不了」的镜像。
COPY install_argos_packages.py .
RUN python3 install_argos_packages.py && rm install_argos_packages.py

COPY server.py .

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
