FROM python:3.12-slim

WORKDIR /app

# 国内网络默认走清华源；海外环境可改为 https://pypi.org/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt

COPY server.py .

EXPOSE 8005

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8005"]
