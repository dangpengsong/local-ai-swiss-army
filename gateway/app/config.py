from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    local_ai_mock: str = "auto"
    asr_url: str = "http://asr:8001"
    tts_url: str = "http://tts:8004"
    nlp_url: str = "http://nlp:8005"
    mtran_url: str = "http://mtran:8989"
    # 文本 AI 的工具能力（function calling）总开关。
    # 请求级还能用 params.tools=false 单独关掉（语音管线就是这么做的）
    nlp_tools: bool = True
    # 对外 OpenAI 兼容接口（/v1/*）的 API Key。
    # 留空 = 不校验，本地私有部署的默认。要暴露到局域网外时再填。
    openai_api_key: str = ""
    # 模型下载源。默认 hf-mirror（国内可用），海外可设为 https://huggingface.co
    hf_endpoint: str = "https://hf-mirror.com"

    asr_port: int = 8001
    tts_port: int = 8004
    nlp_port: int = 8005

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
