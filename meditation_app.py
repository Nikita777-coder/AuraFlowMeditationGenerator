import os
import wave
import struct
import math
import uuid
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, jsonify, request
from pydub import AudioSegment
from yandex_cloud_ml_sdk import YCloudML
from speechkit import model_repository, configure_credentials, creds
from concurrent.futures import ThreadPoolExecutor
import boto3
import re

load_dotenv()

YANDEX_STORAGE_ACCESS_KEY = os.getenv("YANDEX_STORAGE_ACCESS_KEY")
YANDEX_STORAGE_SECRET_KEY = os.getenv("YANDEX_STORAGE_SECRET_KEY")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
GENERATE_MEDITATION_TEXT_PROMPT = os.getenv("GENERATE_MEDITATION_TEXT_PROMPT")
GENERATE_MEDITATION_TEXT_SYSTEM_ROLE_TEXT = os.getenv("GENERATE_MEDITATION_TEXT_PROMPT")
YANDEX_STORAGE_FOLDER_ID = os.getenv("YANDEX_STORAGE_FOLDER_ID")
YANDEX_CLOUD_ML_AUTH = os.getenv("YANDEX_CLOUD_ML_AUTH")
PROMPT_PROCESSING_PROMPT = os.getenv("PROMPT_PROCESSING_PROMPT")
PROMPT_PROCESSING_SYSTEM_ROLE_TEXT = os.getenv("PROMPT_PROCESSING_SYSTEM_ROLE_TEXT")
YANDEX_CLOUD_BUCKET = os.getenv("YANDEX_CLOUD_BUCKET")
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET")

configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key=YANDEX_API_KEY
    )
)


def generate_meditation_text(duration_minutes, meditation_topic):
    prompt = (
            GENERATE_MEDITATION_TEXT_PROMPT %
            (
                duration_minutes,
                meditation_topic
            )
    )
    messages = [
        {
            "role": "system",
            "text": GENERATE_MEDITATION_TEXT_SYSTEM_ROLE_TEXT,
        },
        {
            "role": "user",
            "text": prompt,
        },
    ]
    sdk = YCloudML(
        folder_id=YANDEX_STORAGE_FOLDER_ID,
        auth=YANDEX_CLOUD_ML_AUTH,
    )
    result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
    if result and hasattr(result, "alternatives") and len(result.alternatives) > 0:
        meditation_text = result.alternatives[0].text
        return meditation_text
    else:
        return "Не удалось получить результат"


def add_tts_markup(text):
    text_with_markup = re.sub(r'\.\s+', '. sil<[300]> ', text)
    text_with_markup = re.sub(r'\!\s+', '! sil<[300]> ', text_with_markup)
    text_with_markup = re.sub(r'\?\s+', '? sil<[300]> ', text_with_markup)
    return text_with_markup


def text_to_speech(text, wav_path='output_meditation.wav', mp3_path='output_meditation.mp3'):
    text_with_markup = add_tts_markup(text)

    model = model_repository.synthesis_model()
    model.voice = 'dasha'
    model.role = 'friendly'
    result = model.synthesize(text_with_markup, raw_format=False)
    result.export(wav_path, 'wav')
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    return mp3_path


def prompt_processing(user_request):
    prompt = (
            PROMPT_PROCESSING_PROMPT %
            (
                user_request
            )
    )
    messages = [
        {
            "role": "system",
            "text": PROMPT_PROCESSING_SYSTEM_ROLE_TEXT,
        },
        {
            "role": "user",
            "text": prompt,
        },
    ]
    sdk = YCloudML(
        folder_id=YANDEX_STORAGE_FOLDER_ID,
        auth=YANDEX_CLOUD_ML_AUTH,
    )
    result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
    if result and hasattr(result, "alternatives") and len(result.alternatives) > 0:
        music_prompt = result.alternatives[0].text.strip()
        return music_prompt
    else:
        return "Не удалось обработать запрос для музыки."


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
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
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


def upload_to_yandex_storage(local_file_path, bucket_name, object_name):
    endpoint_url = "https://storage.yandexcloud.net"
    access_key = YANDEX_STORAGE_ACCESS_KEY
    secret_key = YANDEX_STORAGE_SECRET_KEY
    if not access_key or not secret_key:
        raise RuntimeError("❌ Не найдены ключи доступа. Убедитесь, что переменные окружения заданы.")
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    s3.upload_file(local_file_path, bucket_name, object_name)
    public_url = f"{endpoint_url}/{bucket_name}/{object_name}"
    return public_url


def process_meditation(duration_minutes, meditation_topic):
    meditation_text = generate_meditation_text(duration_minutes, meditation_topic)
    return text_to_speech(meditation_text)


def process_melody(melody_request, duration_minutes):
    normalized_keywords = prompt_processing(melody_request)
    return generate_audio_output_stereo(normalized_keywords, duration_minutes)


def validate_auth_token(token):
    tok = token.split("Bearer ")

    if len(tok) < 2 or tok[1] != AUTH_JWT_SECRET:
        raise RuntimeError("invalid token")

    return tok[1]


app = Flask(__name__)


@app.route('/generate_meditation', methods=['POST'])
def generate_meditation_with_melody():
    validate_auth_token(request.headers.get('Authorization'))

    data = request.get_json()
    duration_minutes = data.get('duration')
    meditation_topic = data.get('topic')
    melody_request = data.get('melody')

    with ThreadPoolExecutor() as executor:
        future_meditation = executor.submit(process_meditation, duration_minutes, meditation_topic)
        future_melody = executor.submit(process_melody, melody_request, duration_minutes)
        meditation_mp3 = future_meditation.result()
        melody_wav = future_melody.result()

    meditation_audio = AudioSegment.from_mp3(meditation_mp3)
    melody_audio = AudioSegment.from_wav(melody_wav)

    combined_audio = melody_audio.overlay(meditation_audio)
    final_output_file = "final_output.mp3"
    combined_audio.export(final_output_file, format="mp3")

    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"audio/meditation_with_melody_{timestamp}_{unique_id}.mp3"
    bucket_name = YANDEX_CLOUD_BUCKET
    public_url = upload_to_yandex_storage(final_output_file, bucket_name, object_name)

    return jsonify({"audio_link": public_url})


if __name__ == '__main__':
    app.run(debug=True)
