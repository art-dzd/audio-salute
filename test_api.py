#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Локальное тестирование API Salute Speech"""

import os
import uuid
import requests
import json
import subprocess
from datetime import datetime
import urllib3
from dotenv import load_dotenv

# Отключаем предупреждения SSL для тестирования
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Загрузка переменных окружения
load_dotenv()
SALUTE_SPEECH_API_KEY = os.getenv("SALUTE_SPEECH_API_KEY")
FFMPEG_BIN = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
TEST_AUDIO = os.path.join(os.getcwd(), "audio_samples", "for_load.mp3")
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def get_salute_token():
    if not SALUTE_SPEECH_API_KEY:
        print("Ошибка: SALUTE_SPEECH_API_KEY не задан в .env!")
        exit(1)
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
        print(f"Токен получен, действителен до {expires_at}")
        return access_token
    except Exception as e:
        print(f"Ошибка получения токена: {e}")
        return None

def prepare_audio(audio_path):
    """Подготовка аудиофайла в формат для API Salute Speech"""
    output_path = os.path.join(TEMP_DIR, f"prepared_{uuid.uuid4()}.pcm")
    
    command = [
        FFMPEG_BIN, 
        "-i", audio_path, 
        "-acodec", "pcm_s16le", 
        "-ac", "1",  # Моно
        "-ar", "16000",  # 16kHz
        "-f", "s16le",  # Формат raw PCM
        output_path
    ]
    
    print(f"Конвертирую аудиофайл...")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Аудиофайл успешно конвертирован")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Ошибка конвертации аудио: {e.stderr.decode()}")
        raise Exception("Ошибка при конвертации аудио")

def transcribe_audio(audio_path, token):
    """Отправка аудиофрагмента на распознавание в API Salute Speech"""
    if not token:
        raise Exception("Не передан токен для Salute Speech API")
    
    print("Читаю аудиофайл...")
    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()
    
    url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "audio/x-pcm;bit=16;rate=16000"
    }
    
    print("Отправляю запрос на распознавание...")
    try:
        response = requests.post(
            url, 
            headers=headers, 
            data=audio_content,
            params={
                "language": "ru-RU",
                "model": "general"
            },
            verify=False
        )
        response.raise_for_status()
        result = response.json()
        
        print("Ответ от API получен:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if "status" in result and result["status"] == 200:
            if "result" in result and len(result["result"]) > 0:
                transcript = result["result"][0]
                print(f"\nРаспознанный текст: {transcript}")
                return transcript
        elif "results" in result and len(result["results"]) > 0:
            transcript = result["results"][0].get("alternatives", [{}])[0].get("transcript", "")
            print(f"\nРаспознанный текст: {transcript}")
            return transcript
            
        print("Текст не распознан")
        return ""
    except Exception as e:
        print(f"Ошибка при распознавании аудио: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Код ответа: {e.response.status_code}")
            print(f"Текст ответа: {e.response.text}")
        raise

def main():
    """Основная функция"""
    print(f"Тестирование API Salute Speech на файле {TEST_AUDIO}")
    
    token = get_salute_token()
    if not token:
        print("Не удалось получить токен. Завершение работы.")
        return
    
    try:
        prepared_audio = prepare_audio(TEST_AUDIO)
        transcript = transcribe_audio(prepared_audio, token)
        os.remove(prepared_audio)
        
        if transcript:
            print("\nТест завершен успешно!")
        else:
            print("\nТест завершен, но текст не был распознан.")
            
    except Exception as e:
        print(f"Произошла ошибка при выполнении теста: {e}")

if __name__ == "__main__":
    main() 