import os
import wave
import struct
import math
import psutil
import uuid
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from yandex_cloud_ml_sdk import YCloudML
from speechkit import model_repository, configure_credentials, creds
import boto3
import re
from threading import Thread
import asyncio

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
_MEDITATION_KEYWORDS = os.getenv("MEDITATION_KEYWORDS")
DEFAULT_MEDITATION_TEXT = os.getenv("DEFAULT_MEDITATION_TEXT")
INTEGRATION_SERVICE_GET_TOKEN_URI = os.getenv("INTEGRATION_SERVICE_GET_TOKEN_URI")
INTEGRATION_SERVICE_OIDC_EMAIL = os.getenv("INTEGRATION_SERVICE_OIDC_EMAIL")

configure_credentials(
    yandex_credentials=creds.YandexCredentials(api_key=YANDEX_API_KEY)
)

STATUS_DIR = "status"
os.makedirs(STATUS_DIR, exist_ok=True)
status_store = {}

def _looks_like_meditation(text: str) -> bool:
    if not text or len(text) < 100:
        return False
    score = sum(1 for kw in _MEDITATION_KEYWORDS if kw in text.lower())
    return score >= 2

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"[RAM] {note} Использование памяти: {mem:.2f} MB")

def save_status(id_, status, url=None):
    data = {
        "status": str(status),
        "url": str(url or ""),
        "wasUsed": "false"
    }
    status_store[id_] = data
    print(status_store)

def get_status(id_):
    print(status_store)
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

def generate_meditation_text(duration_minutes, meditation_topic):
    prompt = GENERATE_MEDITATION_TEXT_PROMPT % (duration_minutes, meditation_topic)
    messages = [
        {"role": "system", "text": GENERATE_MEDITATION_TEXT_SYSTEM_ROLE_TEXT},
        {"role": "user", "text": prompt},
    ]
    print_memory_usage("До генерации текста")
    try:
        sdk = YCloudML(folder_id=YANDEX_STORAGE_FOLDER_ID, auth=YANDEX_CLOUD_ML_AUTH)
        result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
        text = result.alternatives[0].text if result and result.alternatives else ""
    except Exception as e:
        print(f"[WARN] Ошибка YandexGPT: {e}")
        text = ""
    print_memory_usage("После генерации текста")
    if not _looks_like_meditation(text):
        print("[INFO] Используем стандартную медитацию из-за нерелевантного запроса.")
        return DEFAULT_MEDITATION_TEXT.strip()
    return text.strip()

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

CHUNK_SIZE_FRAMES = 44100

def load_wave_stereo_stream(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Файл '{filename}' не найден.")
    wf = wave.open(filename, 'rb')
    num_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    if num_channels != 2 or sample_width != 2:
        raise ValueError("Ожидается стерео 16-бит WAV (2 канала, 16 бит).")
    return wf, sample_rate

def read_chunk(wf, chunk_size=CHUNK_SIZE_FRAMES):
    raw_data = wf.readframes(chunk_size)
    samples = []
    for i in range(0, len(raw_data), 4):
        frame = raw_data[i:i+4]
        if len(frame) < 4:
            break
        left, right = struct.unpack('<hh', frame)
        samples.append((left, right))
    return samples

def write_chunk(wf, samples_chunk):
    frames = b''.join(struct.pack('<hh', L, R) for (L, R) in samples_chunk)
    wf.writeframes(frames)

def resample_chunk(samples_chunk, in_sr, out_sr):
    if in_sr == out_sr:
        return samples_chunk
    n_in = len(samples_chunk)
    if n_in == 0:
        return []
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
        left_floor, right_floor = samples_chunk[pos_floor]
        left_ceil, right_ceil = samples_chunk[pos_ceil]
        left_out = left_floor + alpha * (left_ceil - left_floor)
        right_out = right_floor + alpha * (right_ceil - right_floor)
        L = int(round(left_out))
        R = int(round(right_out))
        L = max(min(L, 32767), -32768)
        R = max(min(R, 32767), -32768)
        samples_out.append((L, R))
    return samples_out

def loop_audio_stream(samples_buffer, total_samples_needed):
    if not samples_buffer:
        return []
    output = []
    idx = 0
    n = len(samples_buffer)
    while len(output) < total_samples_needed:
        output.append(samples_buffer[idx])
        idx += 1
        if idx >= n:
            idx = 0
    return output[:total_samples_needed]

def mix_chunks(*chunks):
    if not chunks:
        return []
    length = min(len(c) for c in chunks)
    mixed = []
    for i in range(length):
        sum_left = sum(chunk[i][0] for chunk in chunks)
        sum_right = sum(chunk[i][1] for chunk in chunks)
        sum_left = max(min(sum_left, 32767), -32768)
        sum_right = max(min(sum_right, 32767), -32768)
        mixed.append((sum_left, sum_right))
    return mixed

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

    streams = []
    sample_rate = None
    desired_duration_sec = duration_minutes * 60
    total_samples_needed = desired_duration_sec * 44100
    temp_files = []

    def open_and_prepare(filepath):
        nonlocal sample_rate
        print_memory_usage(f"До открытия {filepath}")
        wf, sr = load_wave_stereo_stream(filepath)
        print_memory_usage(f"После открытия {filepath}")
        if sample_rate is None:
            sample_rate = sr
            return wf
        if sr != sample_rate:
            print(f"Пересемплирование {filepath} с {sr} на {sample_rate}")
            samples = []
            while True:
                chunk = read_chunk(wf)
                if not chunk:
                    break
                samples.extend(chunk)
            wf.close()
            resampled = resample_stereo(samples, sr, sample_rate)
            temp_filename = f"temp_resampled_{uuid.uuid4().hex}.wav"
            temp_files.append(temp_filename)
            with wave.open(temp_filename, 'wb') as temp_wav:
                temp_wav.setnchannels(2)
                temp_wav.setsampwidth(2)
                temp_wav.setframerate(sample_rate)
                for L, R in resampled:
                    temp_wav.writeframes(struct.pack('<hh', L, R))
            wf_new, _ = load_wave_stereo_stream(temp_filename)
            return wf_new
        else:
            return wf

    if mood == "calm":
        streams.append(open_and_prepare(calm_melody_file))
    elif mood == "energetic":
        streams.append(open_and_prepare(energetic_melody_file))

    for ns in nature_sounds:
        filepath = nature_files.get(ns)
        if filepath:
            streams.append(open_and_prepare(filepath))

    if not streams:
        print("[INFO] Не удалось распознать запрос – используем стандартную спокойную мелодию.")
        streams.append(open_and_prepare(calm_melody_file))

    output_wav = wave.open(output_file, 'wb')
    output_wav.setnchannels(2)
    output_wav.setsampwidth(2)
    output_wav.setframerate(sample_rate)

    print_memory_usage("До обработки аудио чанками")

    samples_written = 0
    buffer = bytearray()

    while samples_written < total_samples_needed:
        chunks = []
        valid_streams = []
        for wf in streams:
            chunk = read_chunk(wf)
            if not chunk:
                wf.rewind()
                chunk = read_chunk(wf)
            if chunk:
                chunks.append(chunk)
                valid_streams.append(wf)
        streams = valid_streams
        if not chunks:
            print("Все потоки закончились. Досрочное завершение генерации.")
            break
        min_len = min(len(c) for c in chunks)
        chunks = [c[:min_len] for c in chunks]
        mixed = mix_chunks(*chunks)
        to_write = mixed[:min(total_samples_needed - samples_written, len(mixed))]
        for (left, right) in to_write:
            buffer.extend(struct.pack('<hh', left, right))
        if len(buffer) >= CHUNK_SIZE_FRAMES * 4:
            output_wav.writeframes(buffer)
            buffer.clear()
        samples_written += len(to_write)

    if buffer:
        output_wav.writeframes(buffer)

    print_memory_usage("После обработки аудио чанками")

    for wf in streams:
        wf.close()
    output_wav.close()

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Удален временный файл: {temp_file}")
            except Exception as e:
                print(f"Не удалось удалить {temp_file}: {e}")

    return output_file

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
    samples = []
    for i in range(0, len(raw_data), 4):
        frame = raw_data[i:i+4]
        left, right = struct.unpack('<hh', frame)
        samples.append((left, right))
    return sample_rate, samples

def loop_audio_stereo(samples, sample_rate, desired_duration_sec):
    total_samples_needed = int(desired_duration_sec * sample_rate)
    output = []
    idx = 0
    n = len(samples)
    while len(output) < total_samples_needed:
        output.append(samples[idx])
        idx += 1
        if idx >= n:
            idx = 0
    return output[:total_samples_needed]

def mix_stereo_audios(*tracks):
    if not tracks:
        return []
    length = len(tracks[0])
    for t in tracks:
        if len(t) != length:
            raise ValueError("Все треки должны иметь одинаковую длину.")
    mixed = []
    for i in range(length):
        sum_left = sum(track[i][0] for track in tracks)
        sum_right = sum(track[i][1] for track in tracks)
        sum_left = max(min(sum_left, 32767), -32768)
        sum_right = max(min(sum_right, 32767), -32768)
        mixed.append((sum_left, sum_right))
    return mixed

def save_wave_stereo(filename, sample_rate, samples):
    if not samples:
        raise ValueError("Нет данных для сохранения.")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)  # стерео
        wf.setsampwidth(2)  # 16 бит
        wf.setframerate(sample_rate)
        for (left, right) in samples:
            data = struct.pack('<hh', left, right)
            wf.writeframesraw(data)

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
        combined_path = f"final_{task_id}.mp3"
        os.system(
            f"ffmpeg -y -i {mel_wav} -i {med_mp3} -filter_complex amix=inputs=2:duration=first:dropout_transition=3 -b:a 64k {combined_path}")
        object_name = f"audio/meditation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id}.mp3"
        url = upload_to_yandex_storage(combined_path, YANDEX_CLOUD_BUCKET, object_name)
        save_status(task_id, "ready", url)
        for f in [med_mp3, f"med_{task_id}.wav", mel_wav, combined_path]:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print(f"Ошибка в процессе task_id={task_id}: {e}")
        save_status(task_id, "error")

app = Flask(__name__)

async def auto_cleanup():
    while True:
        await asyncio.sleep(60)
        print(status_store)
        for key in list(status_store.keys()):
            val = status_store.get(key)
            if val and ((val.get("status") == "ready" and val.get("wasUsed") == "true") or (val.get("status") == "error")):
                del status_store[key]

@app.route('/generate_meditation', methods=['POST'])
def generate():
    check_access(request.headers)
    data = request.get_json()
    task_id = uuid.uuid4().hex
    print(data)
    save_status(task_id, "processing")
    Thread(target=process_all, args=(task_id, data['duration'], data['topic'], data['melody'])).start()
    return jsonify(task_id)

@app.route('/status/<task_id>', methods=['GET'])
def status(task_id):
    check_access(request.headers)
    print_memory_usage("Использовано RAM")
    return jsonify(get_status(task_id))

def check_access(headers):
    token = headers.get('Authorization').split("Bearer ")[-1]
    date = headers.get('X-Token-Date')
    params = {
        "email": INTEGRATION_SERVICE_OIDC_EMAIL,
        "date": date
    }
    print(INTEGRATION_SERVICE_GET_TOKEN_URI)
    server_token = requests.get(INTEGRATION_SERVICE_GET_TOKEN_URI, headers=headers, params=params)
    print(server_token.request.url)
    if (token != server_token.text):
        raise RuntimeError("access deny")

if __name__ == '__main__':
    import threading
    threading.Thread(target=lambda: asyncio.run(auto_cleanup()), daemon=True).start()
    app.run(host='127.0.0.1', port=8090)
