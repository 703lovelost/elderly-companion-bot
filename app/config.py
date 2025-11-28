import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    chunk_duration_sec: float = 0.5
    silence_threshold: float = 0.02
    min_silence_duration_sec: float = 1.0


@dataclass
class ModelConfig:
    model_id: str = os.getenv("WHISPER_MODEL_PATH", "./models/whisper-podlodka-turbo")
    language: str | None = "russian"
    task: str = "transcribe"
    device_map: str = "auto"
    torch_dtype: str = "auto"


@dataclass
class TTSConfig:
    language: str = "ru"
    model_id: str = "v5_ru"
    speaker: str = os.getenv("SILERO_TTS_SPEAKER", "eugene")
    sample_rate: int = 48_000
    local_file: str = os.getenv(
        "SILERO_TTS_CHECKPOINT",
        "./models/silero-tts-ru-v5/v5_ru.pt",
    )


@dataclass
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")