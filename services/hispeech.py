import os
import time
import httpx


API_URL = "https://api.hispeech.ai/api/v1/transcriptions/upload"


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    api_key = os.getenv("HISPEECH_API_KEY", "")
    if not api_key:
        raise ValueError("HISPEECH_API_KEY is not set in .env")

    cost_per_min = float(os.getenv("HISPEECH_COST_PER_MINUTE", "0"))
    start = time.time()

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            API_URL,
            headers={"x-auth-token": api_key},
            data={"wait_for_result": "true"},
            files={"file": (filename, audio_bytes, "audio/mpeg")},
        )
        response.raise_for_status()
        data = response.json()

    if str(data.get("success")) != "true":
        raise ValueError(f"HiSpeech error: {data.get('error', 'unknown error')}")

    latency = time.time() - start
    text = data.get("transcription", "")

    return {
        "text": text.strip(),
        "latency": round(latency, 2),
        "cost": round((latency / 60) * cost_per_min, 6),
        "model": "HiSpeech",
    }
