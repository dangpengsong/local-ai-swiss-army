FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预装 argos 常用语言包（en<->zh/ja/ko/fr/de）
# 若构建时网络不可达则跳过，运行时翻译会返回提示
RUN python3 -c "\
from argostranslate.package import get_available_packages, install_from_path;\
pairs=[('en','zh'),('zh','en'),('en','ja'),('ja','en')];\
pkgs=list(get_available_packages());\
[install_from_path(str(p.download())) for s,t in pairs for p in pkgs if p.from_code==s and p.to_code==t];\
print('done')" || echo "WARN: argos 语言包下载跳过（构建时无网络），运行时翻译功能可能受限"

COPY server.py .

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
