from dataclasses import dataclass
import os

from src.env_loader import load_local_env_file

load_local_env_file()


@dataclass
class Settings:
    stt_model: str = os.getenv('STT_MODEL', 'base')
    tts_model: str = os.getenv('TTS_MODEL', 'pyttsx3')
    lang: str = os.getenv('LANG', 'en')
