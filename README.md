<br />
<p align="center">
  <a href="https://github.com/703lovelost/elderly-companion-bot">
    <img src="./docs_src/preview.png" alt="Logo" style="width: 500px; height: auto;">
  </a>

  <h3 align="center">Голосовой собеседник для пожилых людей</h3>
</p>

<p align="center">
  <a href="https://github.com/Pozovi23/"><b>Глеб Жигалов</b></a> · <a href="https://github.com/svetlana-fisher/"><b>Светлана Рыбинцева</b></a> · <a href="https://github.com/703lovelost/"><b>Алексей Спиркин</b></a>
  <br />
  Институт интеллектуальной робототехники
  <br />
  Новосибирский государственный университет
</p>

## Описание проекта

Telegram-бот, который принимает голосовые сообщения, распознает речь пользователя и отвечает в формате голосовых сообщений с расшифровкой.

Проект заточен под **бережное, эмпатичное общение с пожилыми людьми**: мягкий тон, поддержка, уважение
и никаких токсичных ответов.

## Основные возможности

- Приём голосовых сообщений в Telegram
- Локальное распознавание речи
- Генерация ответа через OpenRouter API
- Озвучка ответа с помощью Silero TTS RU v5
- Возврат ответа пользователю в формате голосового сообщения 
- Память о диалоге для каждого пользователя

## Технологический стек

- **Python 3.10+**
- **[python-telegram-bot 20+](https://python-telegram-bot.org/)** для инициализации бота
- **[whisper-podlodka-turbo](https://huggingface.co/bond005/whisper-podlodka-turbo)** для транскрибации речи пользователя
- **[Silero TTS ru_v5](https://github.com/snakers4/silero-models)** для синтеза русской речи
- **[OpenRouter](https://openrouter.ai/)** в качестве шлюза к языковой модели

## Установка

1. Склонируйте репозиторий и установите зависимости:

```
git clone https://github.com/703lovelost/elderly-companion-bot
cd elderly-companion-bot
pip install --upgrade pip
pip install -r requirements.txt
```

2. Для генерации голосовых сообщений требуется наличие пакета `ffmpeg`:

```
sudo apt update
sudo apt install ffmpeg
```

3. Загрузите модели в каталог `models`. Инструкция по установке моделей приложена <a href="./models/README.md">здесь</a>.

## Запуск

```
python3 main.py
```
