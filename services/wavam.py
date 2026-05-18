import os
import io
import time
import json
import base64
import random
import string
import asyncio
import numpy as np
import httpx
from pydub import AudioSegment

STREAM_URL  = "http://wav.am:21478/transcribe_stream/"
PUNCTUATE_URL = "http://wav.am:21478/punctuate/"
CHUNK_SAMPLES = 17792   # ~1.112 seconds at 16 kHz
TARGET_SR     = 16000


def _load_chunks(audio_bytes: bytes, filename: str) -> list:
    fmt = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp3"
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return [samples[i: i + CHUNK_SAMPLES].tolist()
            for i in range(0, len(samples), CHUNK_SAMPLES)]


def _to_base64(floats: list) -> str:
    arr = np.array(floats, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def _stream_id() -> str:
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))
    return f"file_{rand}"


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    token      = os.getenv("WAVAM_API_KEY", "")
    project_id = os.getenv("WAVAM_PROJECT_ID", "")

    if not token:
        raise ValueError("WAVAM_API_KEY is not set in .env")
    if not project_id:
        raise ValueError("WAVAM_PROJECT_ID is not set in .env")

    cost_per_min = float(os.getenv("WAVAM_COST_PER_MINUTE", "0"))
    start      = time.time()
    stream_id  = _stream_id()
    chunks     = _load_chunks(audio_bytes, filename)
    full_text  = ""

    headers = {"Content-Type": "application/json", "Authorization": token}

    async with httpx.AsyncClient(timeout=300) as client:
        for chunk in chunks:
            r = await client.post(
                STREAM_URL,
                headers=headers,
                json={
                    "project_id": project_id,
                    "stream_id":  stream_id,
                    "language":   "hy",
                    "stream":     _to_base64(chunk),
                },
            )
            r.raise_for_status()
            result = r.json()
            if isinstance(result, str):
                result = json.loads(result)
            full_text += result.get("text", "")

        # Punctuate final text
        if full_text.strip():
            pr = await client.post(
                PUNCTUATE_URL,
                headers=headers,
                json={
                    "text":       full_text,
                    "language":   "hy",
                    "id":         stream_id,
                    "project_id": project_id,
                },
            )
            if pr.status_code == 200:
                full_text = pr.json().get("text", full_text)

    latency = time.time() - start
    return {
        "text":    full_text.strip(),
        "latency": round(latency, 2),
        "cost":    round((latency / 60) * cost_per_min, 6),
        "model":   "wav.am",
    }
