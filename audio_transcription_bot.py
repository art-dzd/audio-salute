#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import logging
import asyncio
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydub import AudioSegment
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import urllib3
import shutil
import re

# Отключаем предупреждения SSL для тестирования
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Настройка логирования
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", 8443))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/audio/webhook")
SALUTE_SPEECH_API_KEY = os.getenv("SALUTE_SPEECH_API_KEY")

# Пути к инструментам и временным файлам
if os.name == "nt":
    FFMPEG_BIN = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
else:
    FFMPEG_BIN = "ffmpeg"
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Переменные для хранения токена Salute Speech API
access_token = None
token_expiration = datetime.now()

def clear_temp_dir():
    for f in os.listdir(TEMP_DIR):
        try:
            file_path = os.path.join(TEMP_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Не удалось удалить {f}: {e}")

def get_salute_token():
    global access_token, token_expiration
    if access_token and token_expiration > datetime.now() + timedelta(minutes=1):
        return access_token
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = {"scope": "SALUTE_SPEECH_PERS"}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {SALUTE_SPEECH_API_KEY}"
    }
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10, verify=False)
        response.raise_for_status()
        response_data = response.json()
        access_token = response_data["access_token"]
        expires_at = datetime.fromtimestamp(response_data["expires_at"] / 1000)
        token_expiration = expires_at
        logger.info(f"Получен токен, действителен до {expires_at}")
        return access_token
    except Exception as e:
        logger.error(f"Ошибка получения токена: {e}")
        return None

async def prepare_audio(audio_path):
    output_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.pcm")
    command = [
        FFMPEG_BIN, "-i", audio_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-f", "s16le", output_path
    ]
    proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"Ошибка конвертации аудио: {stderr.decode()}")
        raise Exception("Ошибка при конвертации аудио")
    return output_path

async def split_audio(audio_path, max_duration_ms=60000):
    try:
        # Предварительная проверка аудиофайла
        try:
            audio = AudioSegment.from_file(audio_path)
            # Проверка атрибутов для выявления проблем до разделения
            _ = audio.sample_width
            _ = audio.frame_rate
            _ = audio.channels
        except Exception as e:
            logger.error(f"Ошибка при открытии аудиофайла {audio_path}: {e}")
            raise ValueError(f"Невозможно открыть аудиофайл: {e}")
            
        if len(audio) <= max_duration_ms:
            return [audio_path]
        chunks = []
        for i, start in enumerate(range(0, len(audio), max_duration_ms)):
            chunk = audio[start:start+max_duration_ms]
            chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}_{uuid.uuid4()}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        return chunks
    except Exception as e:
        logger.error(f"Ошибка при разделении аудио: {e}")
        raise

def transcribe_audio_chunk(audio_path):
    token = get_salute_token()
    if not token:
        raise Exception("Не удалось получить токен для Salute Speech API")
    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()
    url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "audio/x-pcm;bit=16;rate=16000"
    }
    try:
        response = requests.post(
            url, headers=headers, data=audio_content,
            params={"language": "ru-RU", "model": "general"}, verify=False
        )
        response.raise_for_status()
        result = response.json()
        if "status" in result and result["status"] == 200:
            if "result" in result and len(result["result"]) > 0:
                return result["result"][0]
        elif "results" in result and len(result["results"]) > 0:
            return result["results"][0].get("alternatives", [{}])[0].get("transcript", "")
        return ""
    except Exception as e:
        logger.error(f"Ошибка при распознавании аудио: {e}")
        raise

async def transcribe_audio(audio_path):
    # 1. Сначала делим исходный файл на куски (wav)
    chunks = await split_audio(audio_path)
    transcripts = []
    failed_chunks = []
    
    # Первый проход по всем фрагментам
    for i, chunk_path in enumerate(chunks):
        try:
            # 2. Каждый кусок конвертируем в .pcm
            prepared_audio = await prepare_audio(chunk_path)
            transcript = transcribe_audio_chunk(prepared_audio)
            transcripts.append(transcript)
            os.remove(chunk_path)
            os.remove(prepared_audio)
        except Exception as e:
            logger.error(f"Ошибка при распознавании фрагмента {i+1}/{len(chunks)} {chunk_path}: {e}")
            failed_chunks.append((i, chunk_path))
    
    # Повторная попытка для неудачных фрагментов
    retry_errors = []
    for i, chunk_path in failed_chunks:
        try:
            logger.info(f"Повторная попытка распознавания фрагмента {i+1}/{len(chunks)}")
            prepared_audio = await prepare_audio(chunk_path)
            transcript = transcribe_audio_chunk(prepared_audio)
            # Вставляем в нужную позицию
            while len(transcripts) <= i:
                transcripts.append("")
            transcripts[i] = transcript
            os.remove(chunk_path)
            os.remove(prepared_audio)
        except Exception as e:
            error_msg = f"Не удалось распознать фрагмент {i+1}/{len(chunks)} даже после повторной попытки: {e}"
            logger.error(error_msg)
            retry_errors.append(error_msg)
            # Удаляем файлы
            try:
                os.remove(chunk_path)
                if os.path.exists(prepared_audio):
                    os.remove(prepared_audio)
            except:
                pass
    
    full_transcript = " ".join(filter(None, transcripts))
    
    # Если были ошибки после повторных попыток, добавляем информацию
    if retry_errors:
        full_transcript += "\n\n[Внимание: Некоторые части аудио не удалось распознать]"
    
    return full_transcript, retry_errors if retry_errors else None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для распознавания речи. Отправь мне аудиофайл, и я переведу его в текст."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Отправь мне аудиосообщение или аудиофайл, и я преобразую его в текст с помощью сервиса распознавания речи Salute от Сбера."
    )

def is_audio_file(file_name):
    audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.wma']
    file_ext = os.path.splitext(file_name.lower())[1]
    return file_ext in audio_extensions

async def handle_voice_or_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = await update.message.reply_text("Получил аудио. Начинаю обработку...")
    try:
        if update.message.voice:
            file_id = update.message.voice.file_id
            file_ext = "ogg"
        elif update.message.audio:
            file_id = update.message.audio.file_id
            file_ext = os.path.splitext(update.message.audio.file_name)[1] if update.message.audio.file_name else "mp3"
            file_ext = file_ext.lstrip(".")
        else:
            await message.edit_text("Ошибка: не могу определить тип аудиофайла.")
            return
        
        file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.{file_ext}")
        
        try:
            # Сначала пробуем стандартный метод
            file = await context.bot.get_file(file_id)
            await file.download_to_drive(file_path)
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Не удалось загрузить файл стандартным методом: {error_message}")
            
            if "file is too big" in error_message.lower():
                await message.edit_text("Файл слишком большой для стандартного API. Пробую альтернативный метод загрузки...")
                # Пробуем альтернативный метод для больших файлов
                success = await download_large_telegram_file(file_id, file_path)
                if not success:
                    await message.edit_text("Не удалось загрузить файл. Файл слишком большой (более 20 МБ). Пожалуйста, отправьте файл меньшего размера или разбейте аудио на части.")
                    return
            else:
                await message.edit_text(f"Произошла ошибка при получении файла: {error_message}")
                return
        
        await message.edit_text("Файл получен. Начинаю распознавание...")
        try:
            transcript, retry_errors = await transcribe_audio(file_path)
            os.remove(file_path)
            # Отправляем результат
            if transcript:
                if len(transcript) <= 4000:
                    await update.message.reply_text(transcript)
                else:
                    transcript_file = os.path.join(TEMP_DIR, f"transcript_{uuid.uuid4()}.txt")
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    await update.message.reply_document(
                        document=open(transcript_file, "rb"), 
                        filename="transcript.txt",
                        caption="Текст распознавания слишком длинный, отправляю как файл."
                    )
                    os.remove(transcript_file)
                
                # Если были ошибки, отправляем дополнительное сообщение
                if retry_errors:
                    error_msg = "При распознавании возникли следующие проблемы:\n" + "\n".join(retry_errors)
                    if len(error_msg) <= 4000:
                        await update.message.reply_text(error_msg)
                    else:
                        error_file = os.path.join(TEMP_DIR, f"errors_{uuid.uuid4()}.txt")
                        with open(error_file, "w", encoding="utf-8") as f:
                            f.write(error_msg)
                        await update.message.reply_document(
                            document=open(error_file, "rb"),
                            filename="errors.txt",
                            caption="Подробная информация об ошибках при распознавании."
                        )
                        os.remove(error_file)
            else:
                await update.message.reply_text("Не удалось распознать речь в аудиофайле.")
            clear_temp_dir()
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио: {e}")
            await update.message.reply_text(f"Произошла ошибка при обработке аудио: {str(e)}")
            clear_temp_dir()
    except Exception as e:
        error_message = str(e)
        logger.error(f"Ошибка при получении файла: {error_message}")
        await message.edit_text(f"Произошла неизвестная ошибка: {error_message}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.document.file_name or not is_audio_file(update.message.document.file_name):
        await update.message.reply_text("Этот документ не является аудиофайлом. Пожалуйста, отправьте аудиофайл.")
        return
    message = await update.message.reply_text("Получил аудиофайл. Начинаю обработку...")
    try:
        file_id = update.message.document.file_id
        file_ext = os.path.splitext(update.message.document.file_name)[1]
        file_ext = file_ext.lstrip(".")
        file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.{file_ext}")
        
        try:
            # Сначала пробуем стандартный метод
            file = await context.bot.get_file(file_id)
            await file.download_to_drive(file_path)
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Не удалось загрузить файл стандартным методом: {error_message}")
            
            if "file is too big" in error_message.lower():
                await message.edit_text("Файл слишком большой для стандартного API. Пробую альтернативный метод загрузки...")
                # Пробуем альтернативный метод для больших файлов
                success = await download_large_telegram_file(file_id, file_path)
                if not success:
                    await message.edit_text("Не удалось загрузить файл. Файл слишком большой (более 20 МБ). Пожалуйста, отправьте файл меньшего размера или разбейте аудио на части.")
                    return
            else:
                await message.edit_text(f"Произошла ошибка при получении файла: {error_message}")
                return
        
        await message.edit_text("Файл получен. Начинаю распознавание...")
        try:
            transcript, retry_errors = await transcribe_audio(file_path)
            os.remove(file_path)
            if transcript:
                if len(transcript) <= 4000:
                    await update.message.reply_text(transcript)
                else:
                    transcript_file = os.path.join(TEMP_DIR, f"transcript_{uuid.uuid4()}.txt")
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    await update.message.reply_document(
                        document=open(transcript_file, "rb"), 
                        filename="transcript.txt",
                        caption="Текст распознавания слишком длинный, отправляю как файл."
                    )
                    os.remove(transcript_file)
                
                # Если были ошибки, отправляем дополнительное сообщение
                if retry_errors:
                    error_msg = "При распознавании возникли следующие проблемы:\n" + "\n".join(retry_errors)
                    if len(error_msg) <= 4000:
                        await update.message.reply_text(error_msg)
                    else:
                        error_file = os.path.join(TEMP_DIR, f"errors_{uuid.uuid4()}.txt")
                        with open(error_file, "w", encoding="utf-8") as f:
                            f.write(error_msg)
                        await update.message.reply_document(
                            document=open(error_file, "rb"),
                            filename="errors.txt",
                            caption="Подробная информация об ошибках при распознавании."
                        )
                        os.remove(error_file)
            else:
                await update.message.reply_text("Не удалось распознать речь в аудиофайле.")
            clear_temp_dir()
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио: {e}")
            await update.message.reply_text(f"Произошла ошибка при обработке аудио: {str(e)}")
            clear_temp_dir()
    except Exception as e:
        error_message = str(e)
        logger.error(f"Ошибка при получении файла: {error_message}")
        await message.edit_text(f"Произошла неизвестная ошибка: {error_message}")

async def download_large_telegram_file(file_id, custom_path):
    """Функция для скачивания больших файлов через прямые URL Telegram"""
    bot_token = TELEGRAM_TOKEN
    file_info_url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"
    
    try:
        response = requests.get(file_info_url)
        response.raise_for_status()
        result = response.json()
        
        if not result.get("ok"):
            error_message = result.get("description", "Неизвестная ошибка")
            if "file is too big" in error_message.lower():
                # Если файл больше 20MB, пробуем другой способ
                try:
                    # Получаем путь напрямую через старый формат URL, который работает для файлов до 2GB
                    # Это не официальный метод, но работает для многих версий Telegram API
                    url = f"https://api.telegram.org/file/bot{bot_token}/old_method/{file_id}"
                    logger.info(f"Пробую скачать большой файл напрямую через URL: {url}")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(custom_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
                except Exception as e:
                    logger.error(f"Ошибка при скачивании большого файла альтернативным методом: {e}")
                    return False
            else:
                raise Exception(error_message)
        
        file_path = result["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(custom_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {e}")
        return False

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_or_audio))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    if SERVER_URL:
        webhook_url = f"{SERVER_URL}{WEBHOOK_PATH}"
        application.run_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            webhook_url=webhook_url,
            url_path=WEBHOOK_PATH,
            drop_pending_updates=True
        )
    else:
        application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main() 