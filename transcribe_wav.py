#!/usr/bin/env python3
"""
Standalone terminal script — transcribe any audio file using wav.am streaming API.

Usage:
  python transcribe_wav.py <audio_file> <token> <project_id> [language]

Example:
  python transcribe_wav.py audio.mp3 YOUR_TOKEN 14386 hy

Languages: hy (Armenian), ru (Russian), en (English)
"""

import sys
import io
import json
import base64
import random
import string
import time

import numpy as np
import httpx
from pydub import AudioSegment

STREAM_URL    = "http://wav.am:21478/transcribe_stream/"
PUNCTUATE_URL = "http://wav.am:21478/punctuate/"
CHUNK_SAMPLES = 17792
TARGET_SR     = 16000


def load_chunks(path: str) -> list:
    fmt = path.rsplit(".", 1)[-1].lower()
    audio = AudioSegment.from_file(path, format=fmt)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return [samples[i: i + CHUNK_SAMPLES].tolist()
            for i in range(0, len(samples), CHUNK_SAMPLES)]


def to_base64(floats: list) -> str:
    return base64.b64encode(np.array(floats, dtype=np.float32).tobytes()).decode("utf-8")


def stream_id() -> str:
    return "file_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=32))


def transcribe(audio_path: str, token: str, project_id: str, language: str = "hy") -> str:
    print(f"\nLoading: {audio_path}")
    chunks = load_chunks(audio_path)
    total  = len(chunks)
    sid    = stream_id()
    print(f"Audio split into {total} chunks — sending to wav.am...\n")

    headers   = {"Content-Type": "application/json", "Authorization": token}
    full_text = ""

    with httpx.Client(timeout=300) as client:
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{total}...", end=" ", flush=True)
            r = client.post(
                STREAM_URL,
                headers=headers,
                json={
                    "project_id": project_id,
                    "stream_id":  sid,
                    "language":   language,
                    "stream":     to_base64(chunk),
                },
            )
            r.raise_for_status()
            result = r.json()
            if isinstance(result, str):
                result = json.loads(result)
            chunk_text = result.get("text", "")
            full_text += chunk_text
            print(chunk_text or "(silent)")

        print("\nPunctuating...")
        if full_text.strip():
            pr = client.post(
                PUNCTUATE_URL,
                headers=headers,
                json={
                    "text":       full_text,
                    "language":   language,
                    "id":         sid,
                    "project_id": project_id,
                },
            )
            if pr.status_code == 200:
                full_text = pr.json().get("text", full_text)

    return full_text.strip()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    audio_path = sys.argv[1]
    token      = sys.argv[2]
    project_id = sys.argv[3]
    language   = sys.argv[4] if len(sys.argv) > 4 else "hy"

    start  = time.time()
    result = transcribe(audio_path, token, project_id, language)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("RESULT:")
    print("="*60)
    print(result)
    print("="*60)
    print(f"Done in {elapsed:.1f}s")

    out = audio_path.rsplit(".", 1)[0] + "_transcription.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Saved to: {out}")
