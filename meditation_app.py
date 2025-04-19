import os
import wave
import struct
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pydub import AudioSegment
from yandex_cloud_ml_sdk import YCloudML
from speechkit import model_repository, configure_credentials, creds
import boto3
import re
from threading import Thread
import logging
import math

status_store = {}  # in-memory fallback instead of redis

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


def save_status(id_, status, url=None):
    data = {
        "status": str(status),
        "url": str(url or ""),
        "wasUsed": "false"
    }
    status_store[id_] = data
    path = os.path.join(STATUS_DIR, f"{id_}.json")
    with open(path, "w") as f:
        json.dump(data, f)


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
    sdk = YCloudML(folder_id=YANDEX_STORAGE_FOLDER_ID, auth=YANDEX_CLOUD_ML_AUTH)
    result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
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
    result = model.synthesize(add_tts_markup(text), raw_format=False)
    result.export(wav_path, 'wav')
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    return mp3_path


def prompt_processing(user_request):
    prompt = PROMPT_PROCESSING_PROMPT % user_request
    messages = [
        {"role": "system", "text": PROMPT_PROCESSING_SYSTEM_ROLE_TEXT},
        {"role": "user", "text": prompt},
    ]
    sdk = YCloudML(folder_id=YANDEX_STORAGE_FOLDER_ID, auth=YANDEX_CLOUD_ML_AUTH)
    result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
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
        sr, data = load_wave_stereo(filepath)
        if sample_rate is None:
            sample_rate = sr
        else:
            if sr != sample_rate:
                data = resample_stereo(data, sr, sample_rate)
        return data

    if mood == "calm":
        data = load_and_resample(calm_melody_file)
        track = loop_audio_stereo(data, sample_rate, desired_duration_sec)
        tracks_to_mix.append(track)
    elif mood == "energetic":
        data = load_and_resample(energetic_melody_file)
        track = loop_audio_stereo(data, sample_rate, desired_duration_sec)
        tracks_to_mix.append(track)
    for ns in nature_sounds:
        filepath = nature_files[ns]
        data = load_and_resample(filepath)
        track = loop_audio_stereo(data, sample_rate, desired_duration_sec)
        tracks_to_mix.append(track)
    if not tracks_to_mix:
        print("Не распознано ни мелодии, ни звуков природы: генерация фонового звука пропущена.")
        return
    if len(tracks_to_mix) == 1:
        final_track = tracks_to_mix[0]
    else:
        final_track = mix_stereo_audios(*tracks_to_mix)
    save_wave_stereo(output_file, sample_rate, final_track)
    return output_file

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
        frame = raw_data[i:i + 4]
        left, right = struct.unpack('<hh', frame)
        samples.append((left, right))
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

def save_wave_stereo(filename, sample_rate, samples):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # моно вместо стерео для экономии
        wf.setsampwidth(2)  # 16 бит
        wf.setframerate(sample_rate)
        for sample in samples:
            left, right = sample if isinstance(sample, tuple) else (sample, sample)
            mono_sample = (left + right) // 2
            wf.writeframesraw(struct.pack('<h', mono_sample))


def upload_to_yandex_storage(local_file_path, bucket_name, object_name):
    s3 = boto3.client(
        's3',
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=YANDEX_STORAGE_ACCESS_KEY,
        aws_secret_access_key=YANDEX_STORAGE_SECRET_KEY
    )
    s3.upload_file(local_file_path, bucket_name, object_name)
    return f"https://storage.yandexcloud.net/{bucket_name}/{object_name}"


def process_all(task_id, duration_minutes, meditation_topic, melody_request):
    try:
        text = generate_meditation_text(duration_minutes, meditation_topic)
        med_mp3 = text_to_speech(text, f"med_{task_id}.wav", f"med_{task_id}.mp3")
        keywords = prompt_processing(melody_request)
        mel_wav = generate_audio_output_stereo(keywords, duration_minutes, f"mel_{task_id}.wav")

        med_audio = AudioSegment.from_mp3(med_mp3)
        mel_audio = AudioSegment.from_wav(mel_wav)
        combined = mel_audio.overlay(med_audio)
        final_path = f"final_{task_id}.mp3"
        combined.export(final_path, format="mp3", bitrate="64k")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"audio/meditation_{timestamp}_{task_id}.mp3"
        url = upload_to_yandex_storage(final_path, YANDEX_CLOUD_BUCKET, object_name)

        save_status(task_id, "ready", url)

        # Удаление временных файлов
        for f in [med_mp3, f"med_{task_id}.wav", mel_wav, final_path]:
            if os.path.exists(f):
                os.remove(f)

    except Exception as e:
        logging.error(f"Error in task {task_id}: {e}")
        save_status(task_id, "error")


import asyncio

app = Flask(__name__)


async def auto_cleanup():
    while True:
        await asyncio.sleep(60)
        for key in list(status_store.keys()):
            val = status_store.get(key)
            if val and val.get("status") == "ready" and val.get("wasUsed") == "true":
                del status_store[key]
                logging.info(f"Auto-removed used meditation: {key}")

        logging.debug("Cleanup cycle complete")


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
    return jsonify(get_status(task_id))


def validate_auth_token(token):
    if not token or not token.startswith("Bearer ") or token.split("Bearer ")[-1] != AUTH_JWT_SECRET:
        raise RuntimeError("invalid token")


if __name__ == '__main__':
    app.run(debug=True)
