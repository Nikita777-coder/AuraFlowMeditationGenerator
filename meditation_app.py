import psutil
import os
import wave
import uuid
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from yandex_cloud_ml_sdk import YCloudML
from speechkit import model_repository, configure_credentials, creds
import boto3
import re
from threading import Thread
import math
import asyncio

status_store = {}

load_dotenv()

YANDEX_STORAGE_ACCESS_KEY = os.getenv("YANDEX_STORAGE_ACCESS_KEY")
YANDEX_STORAGE_SECRET_KEY = os.getenv("YANDEX_STORAGE_SECRET_KEY")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
GENERATE_MEDITATION_TEXT_PROMPT = os.getenv("GENERATE_MEDITATION_TEXT_PROMPT")
GENERATE_MEDITATION_TEXT_SYSTEM_ROLE_TEXT = os.getenv("GENERATE_MEDITATION_TEXT_SYSTEM_ROLE_TEXT")
YANDEX_STORAGE_FOLDER_ID = os.getenv("YANDEX_STORAGE_FOLDER_ID")
YANDEX_CLOUD_ML_AUTH = os.getenv("YANDEX_CLOUD_ML_AUTH")
PROMPT_PROCESSING_PROMPT = os.getenv("PROMPT_PROCESSING_PROMPT")
PROMPT_PROCESSING_SYSTEM_ROLE_TEXT = os.getenv("PROMPT_PROCESSING_SYSTEM_ROLE_TEXT")
YANDEX_CLOUD_BUCKET = os.getenv("YANDEX_CLOUD_BUCKET")
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET")

configure_credentials(
    yandex_credentials=creds.YandexCredentials(api_key=YANDEX_API_KEY)
)

STATUS_DIR = "status"
os.makedirs(STATUS_DIR, exist_ok=True)

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # в МБ
    print(f"[RAM] {note} Использование памяти: {mem:.2f} MB")

def save_status(id_, status, url=None):
    data = {
        "status": str(status),
        "url": str(url or ""),
        "wasUsed": "false"
    }
    status_store[id_] = data


def get_status(id_):
    data = status_store.get(id_)
    if data:
        if data.get("status") == "ready":
            data["wasUsed"] = "true"
        return data

    path = os.path.join(STATUS_DIR, f"{id_}.json")
    if not os.path.exists(path):
        return {"status": "not_found"}
    with open(path) as f:
        data = json.load(f)
    if data.get("status") == "ready":
        os.remove(path)
    return data

    path = os.path.join(STATUS_DIR, f"{id_}.json")
    if not os.path.exists(path):
        return {"status": "not_found"}
    with open(path) as f:
        data = json.load(f)
    if data.get("status") == "ready":
        os.remove(path)
    return data


def generate_meditation_text(duration_minutes, meditation_topic):
    prompt = GENERATE_MEDITATION_TEXT_PROMPT % (duration_minutes, meditation_topic)
    messages = [
        {"role": "system", "text": GENERATE_MEDITATION_TEXT_SYSTEM_ROLE_TEXT},
        {"role": "user", "text": prompt},
    ]

    print_memory_usage("До генерации текста")
    sdk = YCloudML(folder_id=YANDEX_STORAGE_FOLDER_ID, auth=YANDEX_CLOUD_ML_AUTH)
    result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
    print_memory_usage("После генерации текста")

    return result.alternatives[0].text if result and result.alternatives else "Не удалось получить результат"


def add_tts_markup(text):
    text = re.sub(r'\.\s+', '. sil<[300]> ', text)
    text = re.sub(r'!\s+', '! sil<[300]> ', text)
    text = re.sub(r'\?\s+', '? sil<[300]> ', text)
    return text


def text_to_speech(text, wav_path='output.wav', mp3_path='output.mp3'):
    model = model_repository.synthesis_model()
    model.voice = 'dasha'
    model.role = 'friendly'

    print_memory_usage("До генерации звука")
    result = model.synthesize(add_tts_markup(text), raw_format=False)
    result.export(wav_path, 'wav')
    os.system(f"ffmpeg -y -i {wav_path} -b:a 64k {mp3_path}")
    print_memory_usage("После генерации звука")

    return mp3_path


def prompt_processing(user_request):
    prompt = PROMPT_PROCESSING_PROMPT % user_request
    messages = [
        {"role": "system", "text": PROMPT_PROCESSING_SYSTEM_ROLE_TEXT},
        {"role": "user", "text": prompt},
    ]

    print_memory_usage("До генерации текста")
    sdk = YCloudML(folder_id=YANDEX_STORAGE_FOLDER_ID, auth=YANDEX_CLOUD_ML_AUTH)
    result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
    print_memory_usage("После генерации текста")

    return result.alternatives[0].text.strip() if result and result.alternatives else "Не удалось обработать запрос"


def generate_audio_output_stereo(normalized_keywords: str, duration_minutes: int,
                                 output_file: str = "static/audio/result_stereo.wav"):
    req_lower = normalized_keywords.lower()

    mood_map = {
        "спокойн": "calm",
        "бодр": "energetic",
        "энерг": "energetic"
    }
    mood = next((mood_map[key] for key in mood_map if key in req_lower), None)

    nature_map = {
        "лес": "forest",
        "море": "sea",
        "волн": "sea",
        "костер": "fire",
        "огонь": "fire",
        "поле": "pole",
        "ноч": "night"
    }
    nature_sounds = [nature_map[key] for key in nature_map if key in req_lower]

    nature_files = {
        "forest": "static/audio/forest.wav",
        "sea": "static/audio/sea.wav",
        "fire": "static/audio/fire.wav",
        "pole": "static/audio/pole.wav",
        "night": "static/audio/night.wav",
    }
    calm_melody_file = "static/audio/calm.wav"
    energetic_melody_file = "static/audio/energetic.wav"

    tracks_to_mix = []
    sample_rate = None
    desired_duration_sec = duration_minutes * 60

    def load_and_resample(filepath):
        nonlocal sample_rate
        print_memory_usage("До загрузки waw файла")
        sr, data = load_wave_stereo(filepath)
        print_memory_usage("После загрузки waw файла")
        if sample_rate is None:
            sample_rate = sr
        else:
            if sr != sample_rate:
                print_memory_usage("До resample_stereo waw файла")
                data = resample_stereo(data, sr, sample_rate)
                print_memory_usage("После resample_stereo waw файла")
        return data

    if mood == "calm":
        print_memory_usage("До calm load_and_resample процесса")
        data = load_and_resample(calm_melody_file)
        print_memory_usage("После calm load_and_resample процесса")

        print_memory_usage("До calm loop_audio_stereo процесса")
        track = loop_audio_stereo(data, sample_rate, desired_duration_sec)
        print_memory_usage("После calm loop_audio_stereo процесса")

        tracks_to_mix.append(track)
    elif mood == "energetic":
        print_memory_usage("До energetic load_and_resample процесса")
        data = load_and_resample(energetic_melody_file)
        print_memory_usage("После energetic load_and_resample процесса")

        print_memory_usage("До energetic loop_audio_stereo процесса")
        track = loop_audio_stereo(data, sample_rate, desired_duration_sec)
        print_memory_usage("После energetic loop_audio_stereo процесса")

        tracks_to_mix.append(track)
    for ns in nature_sounds:
        filepath = nature_files[ns]

        print_memory_usage("До filepath load_and_resample процесса")
        data = load_and_resample(filepath)
        print_memory_usage("После filepath load_and_resample процесса")

        print_memory_usage("До filepath loop_audio_stereo процесса")
        track = loop_audio_stereo(data, sample_rate, desired_duration_sec)
        print_memory_usage("После filepath loop_audio_stereo процесса")

        tracks_to_mix.append(track)
    if not tracks_to_mix:
        print("Не распознано ни мелодии, ни звуков природы: генерация фонового звука пропущена.")
        return
    if len(tracks_to_mix) == 1:
        final_track = tracks_to_mix[0]
    else:
        print_memory_usage("До mix_stereo_audios")
        final_track = mix_stereo_audios(*tracks_to_mix)
        print_memory_usage("После mix_stereo_audios")

    print_memory_usage("До save_wave_stereo output_file")
    save_wave_stereo(output_file, sample_rate, final_track)
    print_memory_usage("После save_wave_stereo output_file")

    return output_file

def mix_stereo_audios(*tracks: np.ndarray) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 2), dtype=np.int16)

    # Проверка на одинаковую длину и формат
    length = tracks[0].shape[0]
    for i, t in enumerate(tracks):
        if t.shape != (length, 2):
            raise ValueError(f"Трек {i} имеет форму {t.shape}, ожидается ({length}, 2)")

    # Начинаем с первого трека
    mixed = tracks[0].astype(np.int32)

    # Поэтапно прибавляем остальные
    for t in tracks[1:]:
        mixed += t.astype(np.int32)

    # Обрезаем значения в диапазоне int16
    mixed = np.clip(mixed, -32768, 32767)

    return mixed.astype(np.int16)


def loop_audio_stereo(samples: np.ndarray, sample_rate: int, desired_duration_sec: int) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("Ожидается массив формата (N, 2) для стерео сэмплов.")

    total_samples_needed = int(sample_rate * desired_duration_sec)
    n = samples.shape[0]

    if n == 0 or total_samples_needed == 0:
        return np.zeros((0, 2), dtype=np.int16)

    # Сколько полных повторов и сколько дополнительных сэмплов
    full_repeats = total_samples_needed // n
    remainder = total_samples_needed % n

    parts = []

    if full_repeats > 0:
        # Повторяем сэмплы без лишнего копирования
        parts.append(np.tile(samples, (full_repeats, 1)))

    if remainder > 0:
        parts.append(samples[:remainder])

    # Объединяем части
    looped = np.vstack(parts) if parts else np.zeros((0, 2), dtype=np.int16)
    return looped


def load_wave_stereo(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Файл '{filename}' не найден.")

    with wave.open(filename, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        if num_channels != 2 or sample_width != 2:
            raise ValueError("Ожидается стерео 16-бит WAV (2 канала, 16 бит).")

        raw_data = wf.readframes(num_frames)

    # Преобразование в numpy массив формата int16, little-endian
    samples = np.frombuffer(raw_data, dtype='<i2')  # < = little-endian, i2 = int16
    samples = samples.reshape(-1, 2)  # каждая пара: (левый, правый)
    return sample_rate, samples


def resample_stereo(samples_in, in_sr, out_sr):
    if in_sr == out_sr:
        return samples_in
    n_in = len(samples_in)
    duration_sec = n_in / in_sr
    n_out = int(round(duration_sec * out_sr))
    if n_out < 1:
        return []
    samples_out = []
    ratio = (n_in - 1) / float(n_out - 1) if n_out > 1 else 1.0
    for i in range(n_out):
        pos = i * ratio
        pos_floor = int(math.floor(pos))
        pos_ceil = min(pos_floor + 1, n_in - 1)
        alpha = pos - pos_floor
        left_floor, right_floor = samples_in[pos_floor]
        left_ceil, right_ceil = samples_in[pos_ceil]
        left_out = left_floor + alpha * (left_ceil - left_floor)
        right_out = right_floor + alpha * (right_ceil - right_floor)
        L = int(round(left_out))
        R = int(round(right_out))
        L = max(min(L, 32767), -32768)
        R = max(min(R, 32767), -32768)
        samples_out.append((L, R))
    return samples_out


def save_wave_stereo(filename, sample_rate, samples: np.ndarray):
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("Ожидается массив формата (N, 2) для стерео сэмплов.")

    # Усредняем каналы для получения моно
    mono = samples.mean(axis=1).astype(np.int16)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # моно
        wf.setsampwidth(2)  # 16 бит
        wf.setframerate(sample_rate)
        wf.writeframes(mono.tobytes())


def upload_to_yandex_storage(local_file_path, bucket_name, object_name):
    s3 = boto3.client(
        's3',
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=YANDEX_STORAGE_ACCESS_KEY,
        aws_secret_access_key=YANDEX_STORAGE_SECRET_KEY
    )

    print_memory_usage("Использовано RAM YandexStorage до генерации")
    s3.upload_file(local_file_path, bucket_name, object_name)
    print_memory_usage("Использовано RAM YandexStorage после генерации")

    return f"https://storage.yandexcloud.net/{bucket_name}/{object_name}"


def process_all(task_id, duration_minutes, meditation_topic, melody_request):
    try:
        text = generate_meditation_text(duration_minutes, meditation_topic)
        med_mp3 = text_to_speech(text, f"med_{task_id}.wav", f"med_{task_id}.mp3")
        keywords = prompt_processing(melody_request)
        mel_wav = generate_audio_output_stereo(keywords, duration_minutes, f"mel_{task_id}.wav")
        print_memory_usage("Использовано RAM после генерации мелодии")

        combined_path = f"final_{task_id}.mp3"
        os.system(
            f"ffmpeg -y -i {mel_wav} -i {med_mp3} -filter_complex amix=inputs=2:duration=first:dropout_transition=3 -b:a 64k {combined_path}")
        final_path = combined_path
        print_memory_usage("Использовано RAM после сохранения файла")
        # экспорт уже выполнен через ffmpeg

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"audio/meditation_{timestamp}_{task_id}.mp3"
        url = upload_to_yandex_storage(final_path, YANDEX_CLOUD_BUCKET, object_name)

        save_status(task_id, "ready", url)

        for f in [med_mp3, f"med_{task_id}.wav", mel_wav, combined_path]:
            if os.path.exists(f):
                os.remove(f)

    except Exception as e:
        save_status(task_id, "error")

app = Flask(__name__)


async def auto_cleanup():
    while True:
        await asyncio.sleep(60)
        for key in list(status_store.keys()):
            val = status_store.get(key)
            if val and val.get("status") == "ready" and val.get("wasUsed") == "true":
                del status_store[key]


@app.route('/generate_meditation', methods=['POST'])
def generate():
    validate_auth_token(request.headers.get('Authorization'))
    data = request.get_json()
    task_id = uuid.uuid4().hex
    save_status(task_id, "processing")
    Thread(target=process_all, args=(task_id, data['duration'], data['topic'], data['melody'])).start()
    return jsonify({"id": task_id})


@app.route('/status/<task_id>', methods=['GET'])
def status(task_id):
    print_memory_usage("Использовано RAM")
    return jsonify(get_status(task_id))


def validate_auth_token(token):
    if not token or not token.startswith("Bearer ") or token.split("Bearer ")[-1] != AUTH_JWT_SECRET:
        raise RuntimeError("invalid token")
