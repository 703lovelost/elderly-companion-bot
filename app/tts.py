import io
import logging
import os

import numpy as np
import torch
from pydub import AudioSegment

from .config import TTSConfig


class SileroTTS:
    def __init__(self, cfg: TTSConfig | None = None):
        self.cfg = cfg or TTSConfig()

        logging.info(f"Загрузка TTS модели: {self.cfg.model_id}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch.set_num_threads(4)

        local_file = self.cfg.local_file
        directory = os.path.dirname(local_file)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.isfile(local_file):
            raise FileNotFoundError(f"Модель не найдена: {local_file}")

        model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        model.to(self.device)
        self.model = model

        logging.info("TTS модель загружена")

    def synthesize(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)

        with torch.no_grad():
            audio = self.model.apply_tts(
                text=text,
                speaker=self.cfg.speaker,
                sample_rate=self.cfg.sample_rate,
            )

        if isinstance(audio, torch.Tensor):
            waveform = audio.detach().cpu().numpy().astype(np.float32)
        else:
            waveform = np.asarray(audio, dtype=np.float32)

        return waveform

    def synthesize_to_ogg_bytes(self, text: str) -> io.BytesIO:
        waveform = self.synthesize(text)

        if waveform.size == 0:
            return io.BytesIO()

        waveform = np.clip(waveform, -1.0, 1.0)
        audio_int16 = (waveform * 32767).astype(np.int16)

        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.cfg.sample_rate,
            sample_width=2,
            channels=1,
        )

        buf = io.BytesIO()

        try:
            audio_segment.export(buf, format="ogg", codec="libopus")
        except Exception:
            audio_segment.export(buf, format="ogg")

        buf.seek(0)
        return buf