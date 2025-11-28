import logging
import os

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .config import ModelConfig


os.environ.setdefault("HF_HUB_OFFLINE", "1")


def is_speech(chunk: np.ndarray, threshold: float) -> bool:
    rms = np.sqrt(np.mean(chunk ** 2))
    return rms > threshold


class WhisperASR:
    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        model_dir = self.cfg.model_id

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Модель не найдена: {model_dir}")

        logging.info(f"Загрузка модели ASR: {model_dir}")

        device = 0 if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        self.asr = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

        logging.info("ASR модель загружена")

    def transcribe_chunk(self, audio: np.ndarray) -> dict:
        generate_kwargs = {"task": self.cfg.task}
        if self.cfg.language:
            generate_kwargs["language"] = self.cfg.language

        result = self.asr(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )
        return result
