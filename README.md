# 本地AI工具箱 (Local AI Swiss Army)

> 来源：公众号「名侦探科男」
>
> 本地运行 12 个 AI 模型，零联网、零 API Key，Docker 一键部署。

## 架构

```
┌─────────────────────────────────────────────┐
│             Web 面板 / CLI (curl)            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          API Gateway (FastAPI) :8000         │
│  /asr  /translate  /ocr  /tts  /nlp         │
│  /pipeline/voice  /pipeline/document         │
│  /models  (模型管理)                         │
└──┬──────┬──────┬──────┬──────┬───────────────┘
   │      │      │      │      │
┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌─▼───┐
│ ASR ││翻译 ││ OCR ││ TTS ││ NLP │
│:8001││:8002││:8003││:8004││:8005│
└─────┘└─────┘└─────┘└─────┘└─────┘
```

## 12 个模型

| 类别 | 模型 | 引擎 | 备注 |
|------|------|------|------|
| 语音识别 | faster-whisper-base | faster-whisper | CTranslate2 CPU 量化 |
| 语音识别 | Fun-ASR | funasr | 阿里达摩院 |
| 语音识别 | Moonshine | moonshine | 超轻量 |
| 翻译 | MTranServer | 独立 Docker | 离线翻译 |
| 翻译 | Argos Translate | argostranslate | 开源离线 |
| 文字识别 | PPOCR-tiny | paddleocr | 飞桨超轻量 |
| 文字识别 | Tesseract | pytesseract | 经典 OCR |
| 语音合成 | Piper | piper-tts | 轻量 TTS |
| 语音合成 | OuteTTS-0.6B | transformers | 小型神经 TTS |
| 语音合成 | OpenAudio S1-Mini | transformers | 开源语音合成 |
| 文本AI | Qwen2.5-0.5B | llama-cpp-python | 通义千问 |
| 文本AI | SmolLM2 | llama-cpp-python | HuggingFace |

---

## 快速开始

### 前置要求

- Docker Desktop（或 Docker Engine + Docker Compose V2）
- 4GB+ 可用磁盘空间（模型下载）
- macOS / Linux / Windows WSL2

### 1. 克隆项目

```bash
git clone https://github.com/k3vi-07/local-ai-swiss-army.git
cd local-ai-swiss-army
```

### 2. 一键安装

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

脚本会自动：
- 创建模型目录
- 复制 `.env` 配置
- 构建 Gateway 镜像

### 3. 启动服务

**Mock 模式（无需模型，秒级启动）：**

```bash
docker compose up -d gateway
```

访问 `http://localhost:8000`，所有服务返回模拟数据。

**完整模式（下载模型 + 启动所有服务）：**

```bash
# 下载可用模型（约 1.6GB）
./scripts/download-models.sh all

# 构建并启动全部服务
docker compose --profile full up -d --build
```

### 4. Web 面板下载模型

也可以通过 Web 面板管理模型：

1. 访问 `http://localhost:8000`
2. 点击「模型管理」标签页
3. 点击「一键下载全部可用模型」或逐个下载
4. 下载完成后重启服务：`docker compose --profile full restart`

---

## API 文档

启动后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

### 核心端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/status` | 服务状态 |
| `GET` | `/models` | 模型列表与下载状态 |
| `POST` | `/models/{id}/download` | 触发模型下载 |
| `POST` | `/asr` | 语音识别 |
| `POST` | `/translate` | 翻译 |
| `POST` | `/ocr` | 文字识别 |
| `POST` | `/tts` | 语音合成 |
| `POST` | `/nlp` | 文本 AI |
| `POST` | `/pipeline/voice` | 语音管线 |
| `POST` | `/pipeline/document` | 文档管线 |
| `POST` | `/pipeline/custom` | 自定义管线 |

### 请求格式

```bash
# 单模型推理
curl -X POST http://localhost:8000/nlp \
  -H "Content-Type: application/json" \
  -d '{"input": "介绍一下你自己", "model": "qwen", "params": {"task": "chat"}}'

# 翻译
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello World", "model": "argos", "params": {"source": "en", "target": "zh"}}'

# 自定义管线
curl -X POST http://localhost:8000/pipeline/custom \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "steps": ["translate:argos", "tts:piper"]}'
```

### 响应格式

```json
{
  "output": "识别/翻译/推理结果",
  "model": "qwen",
  "latency_ms": 123
}
```

---

## 环境变量

在 `.env` 文件中配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_AI_MOCK` | `auto` | `auto` 自动检测 / `1` 强制 Mock / `0` 强制真实 |
| `ASR_URL` | `http://asr:8001` | ASR 服务地址 |
| `TRANSLATE_URL` | `http://translate:8002` | 翻译服务地址 |
| `OCR_URL` | `http://ocr:8003` | OCR 服务地址 |
| `TTS_URL` | `http://tts:8004` | TTS 服务地址 |
| `NLP_URL` | `http://nlp:8005` | NLP 服务地址 |
| `MTRAN_URL` | `http://mtran:8989` | MTranServer 独立翻译容器地址 |

---

## Docker Compose Profile 说明

| 启动方式 | 命令 | 说明 |
|----------|------|------|
| 仅 Gateway | `docker compose up -d` | Mock 模式，单容器 |
| 全部服务 | `docker compose --profile full up -d` | 包含所有模型服务 |

---

## 演示

```bash
# 运行完整演示脚本
chmod +x scripts/demo.sh
./scripts/demo.sh http://localhost:8000
```

---

## 文件结构

```
local-ai-swiss-army/
├── docker-compose.yml          # 容器编排
├── .env.example                # 环境变量模板
├── .gitignore
├── README.md                   # 本文档
├── gateway/                    # API 网关
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI 入口 + 全部路由 + 模型管理
│       ├── config.py           # 配置管理
│       ├── mock.py             # Mock 输出生成器
│       ├── adapters/           # 5 个模型适配器
│       └── pipelines/          # 管线（语音/文档/自定义）
├── services/                   # 5 个模型服务
│   ├── asr/                    # 语音识别 (faster-whisper)
│   ├── translate/              # 翻译 (argos-translate)
│   ├── ocr/                    # 文字识别 (tesseract + paddleocr)
│   ├── tts/                    # 语音合成 (piper)
│   └── nlp/                    # 文本AI (llama-cpp-python)
├── web/
│   └── index.html              # Web 面板（单文件 SPA）
└── scripts/
    ├── setup.sh                # 一键安装
    ├── demo.sh                 # API 演示
    └── download-models.sh      # 模型下载
```

---

## 开发指南

### 添加新模型

1. 在对应 `services/<name>/server.py` 中添加推理逻辑
2. 在 `gateway/app/adapters/` 对应适配器的 `SUPPORTED_MODELS` 中注册
3. 在 `gateway/app/main.py` 的 `MODEL_REGISTRY` 中添加下载配置

### 添加新管线

1. 在 `gateway/app/pipelines/` 下创建文件
2. 使用 `@register_pipeline("name")` 装饰器
3. 在 `main.py` 中添加路由

---

## 常见问题

**Q: 容器内无法下载模型？**
A: 国内网络可能需要配置代理或使用镜像。运行 `./scripts/download-models.sh` 在宿主机下载。

**Q: 构建很慢？**
A: NLP 和 ASR 服务需要编译 C++ 依赖（llama-cpp-python、faster-whisper），首次构建约 5-10 分钟。

**Q: 内存不足？**
A: Qwen2.5-0.5B 约 500MB 内存，SmolLM2 约 1.2GB。可只下载需要的模型。

## License

MIT
