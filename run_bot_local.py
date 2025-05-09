#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Скрипт для запуска бота"""

import logging
from dotenv import load_dotenv
import audio_transcription_bot

# Настройка логгера
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Загружаем переменные окружения из .env
    load_dotenv()
    
    logger.info("Запускаю бота для распознавания аудио...")
    
    # Запускаем основной скрипт
    audio_transcription_bot.main() 