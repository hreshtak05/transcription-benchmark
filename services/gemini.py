import os
import time
import asyncio
import tempfile
import pathlib
import google.generativeai as genai
from mutagen import File as MutagenFile


TOKENS_PER_SECOND = 25       # Gemini audio: ~25 tokens/sec
COST_PER_1M_TOKENS = 0.10    # Gemini 2.0 Flash audio input pricing


def _get_duration(path: str) -> float:
    try:
        audio = MutagenFile(path)
        return audio.info.length if audio else 60.0
    except Exception:
        return 60.0


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=api_key)

    ext = pathlib.Path(filename).suffix or ".mp3"
    tmp_path = None
    gemini_file = None
    start = time.time()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        duration = _get_duration(tmp_path)
        gemini_file = genai.upload_file(tmp_path, display_name=filename)

        for _ in range(60):
            gemini_file = genai.get_file(gemini_file.name)
            if gemini_file.state.name != "PROCESSING":
                break
            await asyncio.sleep(2)

        if gemini_file.state.name == "FAILED":
            raise ValueError("Gemini could not process this audio file")

        model = genai.GenerativeModel("gemini-2.5-pro")
        response = await model.generate_content_async([
            gemini_file,
            "Transcribe this audio file accurately. Return ONLY the transcription text — no headers, no commentary, no timestamps.",
        ])

        latency = time.time() - start
        tokens_used = duration * TOKENS_PER_SECOND
        cost = (tokens_used / 1_000_000) * COST_PER_1M_TOKENS

        return {
            "text": response.text.strip(),
            "latency": round(latency, 2),
            "cost": round(cost, 6),
            "duration_seconds": round(duration, 1),
            "model": "gemini-2.5-pro",
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if gemini_file:
            try:
                genai.delete_file(gemini_file.name)
            except Exception:
                pass
