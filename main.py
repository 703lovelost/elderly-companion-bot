import logging

from app.config import Settings
from app.bot import build_application


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    settings = Settings()

    if not settings.telegram_token or not settings.openrouter_api_key:
        raise RuntimeError(
            "Переменные TELEGRAM_TOKEN и OPENROUTER_API_KEY должны быть заданы в .env"
        )

    application = build_application(settings)

    logging.info("Бот запущен")
    application.run_polling()


if __name__ == "__main__":
    main()