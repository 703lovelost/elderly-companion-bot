import io
import logging

import numpy as np
import requests
from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .asr import WhisperASR, is_speech
from .config import AudioConfig, ModelConfig, Settings, TTSConfig
from .tts import SileroTTS


class ElderlyCompanionBot:
    def __init__(
        self,
        api_key: str,
        model: str = "x-ai/grok-4.1-fast:free",
        audio_cfg: AudioConfig | None = None,
        asr_cfg: ModelConfig | None = None,
        tts_cfg: TTSConfig | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.conversations: dict[int, list[dict[str, str]]] = {}

        self.system_prompt = (
            """Ты — добрый, терпеливый и заботливый собеседник для пожилых людей. Тебя зовут Сэм.
Твоя задача — поддерживать простую, теплую беседу, проявляя эмпатию и искренний интерес к словам пользователя.
Тон: мягкий, уважительный, спокойный, ободряющий. Говори кратко и естественно, как в обычном разговоре с другом — без лишних деталей.
Строго запрещено: грубость, резкость, нетерпение, пожелания смерти, негативные или пессимистичные суждения, неприемлемый язык. Матерные слова не упоминай, даже в завуалированном виде.
На рассказы: реагируй с интересом, используя короткие фразы вроде "Как интересно!", "Расскажите подробнее" или "Я слушаю".
На вопросы: давай четкие, простые, доброжелательные ответы. Если вопрос сложный или за пределами твоих знаний, мягко уходи: "К сожалению, не могу точно сказать, но мне интересно ваше мнение".
Если негатив продолжается, плавно смени тему на что-то позитивное.
На грустные или тревожные слова: покажи сочувствие кратко, например: "Я понимаю, это грустно", "Вы сильный человек" или "Я с вами".
Стратегия: поощряй общение, задавай простые вопросы о воспоминаниях, семье, увлечениях. Держи ответы лаконичными, чтобы не утомлять.
Всегда отвечай на русском языке. ЧИСЛИТЕЛЬНЫЕ ПИШИ СЛОВАМИ"""
        )

        self.audio_cfg = audio_cfg or AudioConfig()
        self.asr = WhisperASR(asr_cfg)
        self.tts = SileroTTS(tts_cfg)

        self.min_silence_chunks = int(
            self.audio_cfg.min_silence_duration_sec / self.audio_cfg.chunk_duration_sec
        )

    def get_conversation(self, user_id: int) -> list[dict[str, str]]:
        if user_id not in self.conversations:
            self.conversations[user_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
        return self.conversations[user_id]

    def send_message(self, user_id: int, user_message: str) -> str:
        messages = self.get_conversation(user_id)
        messages.append({"role": "user", "content": user_message})

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            assistant_message = data["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            logging.error(f"Ошибка API: {e}")
            return "Извините, произошла ошибка. Попробуйте еще раз."

    def generate_tts_voice(self, text: str) -> io.BytesIO | None:
        try:
            buf = self.tts.synthesize_to_ogg_bytes(text)
            if buf.getbuffer().nbytes == 0:
                return None
            return buf
        except Exception as e:
            logging.error(f"Ошибка TTS: {e}")
            return None


def build_application(settings: Settings) -> Application:
    bot_instance = ElderlyCompanionBot(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
    )

    application = ApplicationBuilder().token(settings.telegram_token).build()

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        user_id = update.effective_user.id
        bot_instance.get_conversation(user_id)

        await update.message.reply_text(
            "Здравствуйте! Я ваш заботливый собеседник. "
            "Отправьте мне голосовое сообщение, и я с радостью отвечу."
        )

    async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.voice:
            return

        user_id = update.effective_user.id
        voice = await update.message.voice.get_file()
        voice_bytes = await voice.download_as_bytearray()

        try:
            audio = AudioSegment.from_file(io.BytesIO(voice_bytes), format="ogg")
            audio = audio.set_frame_rate(bot_instance.audio_cfg.sample_rate).set_channels(1).set_sample_width(2)
            samples = (
                np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            )
        except Exception as e:
            logging.error(f"Ошибка конвертации аудио: {e}")
            await update.message.reply_text(
                "Извините, не удалось обработать голосовое сообщение."
            )
            return

        audio_buffer = np.array([], dtype=np.float32)
        consecutive_silence = 0
        chunk_size = int(
            bot_instance.audio_cfg.sample_rate
            * bot_instance.audio_cfg.chunk_duration_sec
        )

        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]

            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            if is_speech(chunk, bot_instance.audio_cfg.silence_threshold):
                audio_buffer = np.concatenate((audio_buffer, chunk))
                consecutive_silence = 0
            else:
                if len(audio_buffer) > 0:
                    consecutive_silence += 1
                    if consecutive_silence >= bot_instance.min_silence_chunks:
                        break

        if len(audio_buffer) == 0:
            await update.message.reply_text("Не услышал речь. Попробуйте снова.")
            return

        result = bot_instance.asr.transcribe_chunk(audio_buffer)
        user_input = result.get("text", "").strip()

        logging.info(f"Расшифрованный текст: {user_input}")

        if not user_input:
            await update.message.reply_text("Не услышал речь. Попробуйте снова.")
            return

        response = bot_instance.send_message(user_id, user_input)
        tts_audio = bot_instance.generate_tts_voice(response)

        if tts_audio is not None:
            caption = response[:1024]
            await update.message.reply_voice(voice=tts_audio, caption=caption)
        else:
            await update.message.reply_text(response)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    return application