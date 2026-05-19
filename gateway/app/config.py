from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    local_ai_mock: str = "auto"
    asr_url: str = "http://asr:8001"
    translate_url: str = "http://translate:8002"
    ocr_url: str = "http://ocr:8003"
    tts_url: str = "http://tts:8004"
    nlp_url: str = "http://nlp:8005"
    mtran_url: str = "http://mtran:8989"

    asr_port: int = 8001
    translate_port: int = 8002
    ocr_port: int = 8003
    tts_port: int = 8004
    nlp_port: int = 8005

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
