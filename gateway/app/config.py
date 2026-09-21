from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    local_ai_mock: str = "auto"
    asr_url: str = "http://asr:8001"
    tts_url: str = "http://tts:8004"
    nlp_url: str = "http://nlp:8005"
    mtran_url: str = "http://mtran:8989"
    # NLP 后端协议：llamacpp = 自建服务的 /infer；openai = 官方 llama.cpp server
    # 的 /v1/chat/completions。docker-compose.gpu.yml 会把它设成 openai
    nlp_backend: str = "llamacpp"
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
