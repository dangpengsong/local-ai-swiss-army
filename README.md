# 本地AI工具箱 (Local AI Swiss Army)

> 来源：公众号「名侦探科男」
>
> 本地运行 9 个 AI 模型，零联网、零 API Key，Docker 一键部署。

## 架构

```
┌─────────────────────────────────────────────┐
│             Web 面板 / CLI (curl)            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          API Gateway (FastAPI) :8000         │
│  /asr  /translate  /tts  /nlp               │
│  /pipeline/voice                             │
│  /models  (模型管理)                         │
└──┬──────┬──────┬──────┬──────────────────────┘
   │      │      │      │
┌──▼──┐┌──▼──┐┌──▼──┐┌─▼───┐
│ ASR ││翻译 ││ TTS ││ NLP │
│:8001││:8002││:8004││:8005│
└─────┘└─────┘└─────┘└─────┘
```

## 9 个模型

| 类别 | 模型 | 引擎 | 备注 |
|------|------|------|------|
| 语音识别 | faster-whisper-base | faster-whisper | CTranslate2 CPU 量化 |
| 语音识别 | Fun-ASR | funasr | ⚠️ 未实现（规划中，UI 已禁用） |
| 语音识别 | Moonshine | moonshine | ⚠️ 未实现（规划中，UI 已禁用） |
| 翻译 | MTranServer | 独立 Docker | 离线翻译（自带 `/imme`，可对接沉浸式翻译） |
| 语音合成 | Piper 小雅 | piper-tts | 默认中文语音（拼音方案） |
| 语音合成 | Piper 华言 | piper-tts | 备选中文语音（espeak 方案，中英混读略好） |
| 语音合成 | OuteTTS-0.6B | transformers | ⚠️ 未实现（规划中，UI 已禁用） |
| 语音合成 | OpenAudio S1-Mini | transformers | ⚠️ 未实现（规划中，UI 已禁用） |
| 文本AI | **Qwen3-4B** | llama-cpp-python | 通义千问，纯 CPU 约 7 tok/s（支持流式，首 token 0.15s） |

> 标注「⚠️ 未实现」的 4 个模型仅在注册表中占位，选择后不会返回真实推理结果。

---

## 快速开始

### 前置要求

- Docker Desktop（或 Docker Engine + Docker Compose V2）
- 12GB+ 可用磁盘空间：模型约 3GB（含 Qwen3-4B 的 2.3GB）+ 镜像约 4GB + 构建缓存约 2.8GB（NLP 服务的 llama-cpp-python 需现场编译）
- macOS / Linux / Windows WSL2

### 联网要求

**运行时完全离线**，但首次部署需要联网拉取以下几类资源。任何一步失败，对应功能都会明确报错而非静默降级：

| 阶段 | 拉取内容 | 失败后果 |
|------|----------|----------|
| 构建镜像 | PyPI 依赖（默认走清华源） | 构建失败 |
| 下载模型 | HuggingFace 权重（脚本自动替换为 hf-mirror） | 模型不可用，模型页显示「可下载」 |
| 首次翻译 | MTranServer 按语言对下载模型到 `models/mtran/` | 该语言对翻译失败 |

海外网络可将各 `Dockerfile` 里的 `PIP_INDEX_URL` 改为 `https://pypi.org/simple`，并把 `.env` 里的 `HF_ENDPOINT` 改为 `https://huggingface.co`（控制容器内的两处下载：TTS 拉取 g2pW 注音模型、Web 面板「模型管理」页下载模型）。

### 版本锁定说明

所有 `requirements.txt` 用 `==` 锁定到实测可用的版本，`docker-compose.yml` 里的外部镜像（mtranserver）也固定了 tag。**这样保证任何人 clone 后构建出的依赖与作者验证过的一致** —— 依赖主版本升级导致的 API 变更（例如 `transformers` 4→5、`numpy` 1→2 这类）不会悄悄影响你的部署。

升级某个包时请整体构建测试后，再同步修改对应 `requirements.txt`。

### 构建源可替换

各 `Dockerfile` 顶部的 `ARG` 可按网络环境覆盖：

| ARG | 默认值 | 用途 |
|-----|--------|------|
| `PIP_INDEX_URL` | 清华 PyPI 镜像 | 所有服务的 pip 依赖 |

```bash
# 海外网络可整体切回官方源
docker compose --profile full build \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple
```

模型下载脚本的源同样可覆盖：`HF_MIRROR`（HuggingFace 镜像）。

```bash
HF_MIRROR=https://huggingface.co ./scripts/download-models.sh all
```

> ⚠️ **两个变量名别混**，它们分属两层：
>
> | 变量 | 作用范围 | 控制什么 |
> |------|----------|----------|
> | `HF_MIRROR` | 宿主机命令行 | `scripts/download-models.sh` 的下载 |
> | `HF_ENDPOINT` | `.env` → 容器内 | TTS 拉 g2pW、Web 面板「模型管理」页下载 |
>
> 命令行下载和面板下载走的是两套独立代码，只改一个不会影响另一个。

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

> 模拟输出的文本统一带 `[模拟]` 前缀，前端同时显示黄色「模拟」徽章。这样即使只部署了部分服务（例如只起了 nlp、mtran 容器没起），也不会把模拟结果误认为真实翻译/识别结果。

**完整模式（下载模型 + 启动所有服务）：**

```bash
# 下载可用模型（约 2.7GB）
./scripts/download-models.sh all

# 构建并启动全部服务
docker compose --profile full up -d --build
```

**GPU 加速（可选，需 NVIDIA 显卡）：**

有 NVIDIA 显卡时，文本 AI 可以走 GPU —— 用官方 llama.cpp CUDA 镜像替换自建的 CPU 版服务，实测**生成速度 2.6 倍**（GTX 960 上 18.7 vs 7.3 tok/s，首 token 142ms）：

```bash
# 服务名 / 端口 / profile 与上面完全一致，只是多一个 -f
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile full up -d
```

不加 `-f docker-compose.gpu.yml` 时行为与原来一字不差 —— 这个覆盖层只替换 `nlp` 一个服务的实现。

几点注意：

- **需要可用的 NVIDIA 驱动**。Docker Desktop for Windows 自带 GPU 支持，无需另装 `nvidia-container-toolkit`；没有显卡就别加这个 `-f`，容器会因请求不到设备而起不来
- **启动后约 30 秒模型才就绪**（2.4GB 权重加载进显存）。这段时间 `/health` 返回 503，gateway 按既有的失败降级策略先返回 Mock 结果，就绪后自动恢复，不需要干预
- **上下文固定 4096，不要调大**：4GB 显存下这已是上限（实测 3786/4096 MiB 已用）。同理 `-b 512 -ub 512` 也不能省，原因写在 `docker-compose.gpu.yml` 的注释里
- 此模式下 `.env` 里的 `NLP_N_THREADS` / `NLP_N_CTX` **不生效**——它们只被自建的 Python 服务读取，官方 server 认的是命令行参数
- 显卡显存小于 4GB 时可能装不下，先用 `docker compose ... logs nlp` 看有没有 `failed to fit params` 之类的报错

### 4. Web 面板下载模型

也可以通过 Web 面板管理模型：

1. 访问 `http://localhost:8000`
2. 点击「模型管理」标签页
3. 点击「一键下载全部可用模型」或逐个下载
4. 下载完成后重启服务：`docker compose --profile full restart`

### 5. 切换 TTS 中文语音

内置两个 Piper 中文语音，默认使用**小雅**：

| 语音 | 方案 | 特点 | 许可 |
|------|------|------|------|
| `zh_CN-xiao_ya-medium` | 拼音（g2pW） | 默认，停顿更自然 | **仅限非商用**（DataBaker 数据集） |
| `zh_CN-huayan-medium` | espeak | 中英混读略好，无需额外资源 | 未标注（源数据集 HuaYan_TTS） |

> ⚠️ 两个内置中文语音均**不适合直接商用**：小雅为非商用许可，华言未标注许可。商用场景请替换为自有授权的语音包或改用其他 TTS 引擎。

全局切换：改 `.env` 里的 `TTS_PIPER_VOICE` 后重启 TTS 服务。

```bash
docker compose up -d --no-build tts
```

单次请求指定（不影响全局默认）：

```bash
curl -X POST http://localhost:8000/tts -H 'Content-Type: application/json' \
  -d '{"input":"你好","model":"piper","params":{"voice":"zh_CN-huayan-medium"}}'
```

> 拼音方案（小雅）依赖 g2pW 注音模型（约 152MB），已随 `./scripts/download-models.sh tts` 一并下载到 `models/tts/g2pW/`；若缺失，服务首次合成时会自动从 `HF_ENDPOINT`（默认 hf-mirror）拉取。

### 6. 对接沉浸式翻译（浏览器扩展）

MTranServer 自带 `/imme` 端点，可直接作为沉浸式翻译的「自定义 API」后端，**无需任何适配代码**。

**第一步，启动翻译服务：**

```bash
docker compose up -d --no-build mtran
```

> mtran 虽属于 `full` profile，但它没有任何 `depends_on`，单独启动是安全的（不像指定 tts 会连带拉起整个 profile）。验证：`curl http://localhost:8989/health` 返回 `{"status":"ok"}`。

**第二步，在沉浸式翻译里配置：**

1. 扩展「设置」→「开发者设置」→ 开启「**Beta 特性**」
2. 「翻译服务」→ 找到「**自定义 API**」→ 展开
3. **API URL** 填：`http://localhost:8989/imme`
4. 建议调整：「每秒最大请求数」拉到 512，「每次请求最大段落数」设为 1
5. 点「测试」，出现绿色即对接成功

**模型下载**：首次翻译某个语言对时自动下载并缓存到 `models/mtran/`（已挂载卷，容器重建不丢失）。英中模型约 50MB，首次请求需等几十秒，之后毫秒级响应。

**语言对限制**：MTranServer 以**英语为中心**，非英语互译需经英语中转——例如中译日 = 中译英 + 英译日，需下载两个模型。**英译中 / 中译英是最佳场景**。

**加密码（可选）**：默认无认证。如需启用，给 compose 里的 mtran 服务加环境变量 `MT_API_TOKEN=你的密钥`，URL 改为 `http://localhost:8989/imme?token=你的密钥`。

> 注：沉浸式翻译走的是 8989 端口（MTranServer），不经过 gateway 的 `/translate`——因为沉浸式翻译要求批量数组格式（`text_list` → `translations[]`），与 gateway 的单条格式不兼容。

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
| `POST` | `/tts` | 语音合成 |
| `POST` | `/nlp` | 文本 AI |
| `POST` | `/nlp/stream` | 文本 AI（流式 SSE） |
| `POST` | `/pipeline/voice` | 语音管线 |
| `POST` | `/pipeline/custom` | 自定义管线 |

### 请求格式

```bash
# 单模型推理
curl -X POST http://localhost:8000/nlp \
  -H "Content-Type: application/json" \
  -d '{"input": "介绍一下你自己", "model": "qwen3", "params": {"task": "chat"}}'

# 翻译
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello World", "model": "mtran", "params": {"source": "en", "target": "zh"}}'

# 自定义管线
curl -X POST http://localhost:8000/pipeline/custom \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "steps": ["translate:mtran", "tts:piper"]}'
```

### 响应格式

```json
{
  "output": "识别/翻译/推理结果",
  "model": "qwen3",
  "latency_ms": 123
}
```

### 流式输出（SSE）

`/nlp/stream` 是 `/nlp` 的流式版本，逐 token 推送。实测首 token 从十余秒降到 0.15 秒
（4B 模型纯 CPU 约 7 tok/s，等完整结果出来要 8 秒以上）。

```bash
curl -N -X POST http://localhost:8000/nlp/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "介绍一下你自己", "model": "qwen3"}'
```

```
data: {"delta": "杭州"}

data: {"delta": "是"}

data: {"done": true, "model": "qwen3", "latency_ms": 8600}
```

错误也走流内（`data: {"error": "..."}`）：HTTP 200 在第一个 token 时就已发出，
之后无法再改状态码。`Mock` 模式同样逐字推送，不会让前端停在加载态。

> 管线（`/pipeline/voice`、`/pipeline/custom`）仍走非流式 —— 下一步需要完整文本才能开始。
> 推理本身是串行的（同一个模型实例不能并发），流式不会让两个请求同时生成。

---

## 环境变量

在 `.env` 文件中配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_AI_MOCK` | `auto` | `auto` 自动检测 / `1` 强制 Mock / `0` 强制真实 |
| `ASR_URL` | `http://asr:8001` | ASR 服务地址 |
| `TTS_URL` | `http://tts:8004` | TTS 服务地址 |
| `NLP_URL` | `http://nlp:8005` | NLP 服务地址 |
| `MTRAN_URL` | `http://mtran:8989` | MTranServer 独立翻译容器地址 |
| `NLP_BACKEND` | `llamacpp` | NLP 后端协议：`llamacpp` = 自建服务的 `/infer`；`openai` = 官方 llama.cpp server 的 `/v1/chat/completions`。GPU 覆盖层会自动设为 `openai`，一般不用手动改 |
| `NLP_N_THREADS` | 物理核心数 | NLP 推理线程数。**不要填逻辑核数**——超线程的两个线程会争抢同一物理核的执行单元，实测 i5-10400（6 核 12 线程）上 12 线程比 6 线程慢 **28 倍**（43 → 1.5 tok/s）。GPU 模式下不生效 |
| `NLP_N_CTX` | `4096` | NLP 上下文长度。KV cache 约 144KB/token（Qwen3-4B，36 层 / 8 个 KV 头），4096 约占 590MB 内存。GPU 模式下不生效 |

---

## Docker Compose Profile 说明

| 启动方式 | 命令 | 说明 |
|----------|------|------|
| 仅 Gateway | `docker compose up -d` | Mock 模式，单容器 |
| 全部服务 | `docker compose --profile full up -d` | 包含所有模型服务（CPU） |
| 全部服务 + GPU | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile full up -d` | 同上，但 NLP 走 CUDA 加速 |

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
├── docker-compose.gpu.yml      # GPU 覆盖层（可选，把 nlp 换成 CUDA 镜像）
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
│       ├── adapters/           # 4 个模型适配器
│       └── pipelines/          # 管线（语音/自定义）
├── services/                   # 3 个模型服务
│   ├── asr/                    # 语音识别 (faster-whisper)
│   ├── tts/                    # 语音合成 (piper)
│   └── nlp/                    # 文本AI (CPU 走 llama-cpp-python；GPU 模式被覆盖层替换)
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

以下 5 处都要改。漏改的后果都是「静默不工作」而非报错，排查成本高：

1. `services/<name>/server.py` —— 推理逻辑
2. `gateway/app/adapters/` 对应适配器的 `SUPPORTED_MODELS` —— 漏掉则接口返回「不支持的模型」
3. `gateway/app/main.py` 的 `MODEL_REGISTRY` —— 下载地址与就绪判定
4. `gateway/app/main.py` 的 `_model_service_urls` —— 漏掉则模型页永远显示「服务未启动」
5. `web/index.html` 的下拉框、`scripts/download-models.sh` —— 否则前端选不到、命令行下不到

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

**Q: 只想重建其中一个服务（如 TTS）？**
A: 不要用 `--profile full` 配合 `up --build` —— 它会重建该 profile 下的**所有**服务（含 asr/nlp 的大依赖，数 GB 下载）。分开两步执行：

```bash
docker compose build tts              # 只构建该服务
docker compose up -d --no-build tts   # 只启动该服务
```

**Q: 内存不足？**
A: Qwen3-4B 常驻约 2.9GB —— Q4_K_M 权重 2.3GB + KV cache 约 590MB（默认 `NLP_N_CTX=4096`）。机器吃紧可调小 `NLP_N_CTX`。

**Q: 模型页显示「服务未启动」，但容器明明在跑？**
A: 三种状态含义不同，模型页用的是**文件是否到位**与**服务是否在跑**两个维度：

| 显示 | 含义 | 处理 |
|------|------|------|
| 🟢 已就绪 | 权重齐 + 服务在跑 | 正常 |
| 🟡 已下载 · 服务未启动 | 权重在，但对应容器没起 | `docker compose --profile full up -d` |
| 🟡 服务在跑 · 模型未就绪 | 由独立容器提供的模型（mtran），服务在跑但自身缺资源 | 见下条 |
| ⚪ 服务未启动 | 由独立容器提供的模型（mtran），容器没起 | `docker compose --profile full up -d` |
| ⚪ 未实现 | 仅注册表占位，尚未实现 | 无需处理 |

**Q: 怎样确认部署是完整的？**
A: 打开 `http://localhost:8000` 的「模型管理」页，已实现且有对应服务的模型应显示 🟢 已就绪。或用命令行：

```bash
curl -s localhost:8000/models | python3 -m json.tool | grep -E '"id"|downloaded|service_running'
```

## 关注公众号

<p align="center">
  <img src="assets/qrcode.png" width="360" alt="公众号二维码">
</p>

<p align="center">关注「名侦探科男」，获取更多本地 AI 部署教程</p>

---

## License

MIT
