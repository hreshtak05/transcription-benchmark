import os
import time
import httpx


def _cfg():
    return {
        "url": os.getenv("HISPEECH_API_URL", ""),
        "key": os.getenv("HISPEECH_API_KEY", ""),
        "cost_per_min": float(os.getenv("HISPEECH_COST_PER_MINUTE", "0")),
    }


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    cfg = _cfg()
    if not cfg["key"]:
        raise ValueError("HISPEECH_API_KEY is not set in .env")
    if not cfg["url"]:
        raise ValueError("HISPEECH_API_URL is not set in .env")

    start = time.time()

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            cfg["url"],
            headers={"Authorization": f"Bearer {cfg['key']}"},
            files={"audio": (filename, audio_bytes, "audio/mpeg")},
        )
        response.raise_for_status()
        data = response.json()

    latency = time.time() - start
    text = data.get("text") or data.get("transcription") or data.get("result") or ""

    return {
        "text": text.strip(),
        "latency": round(latency, 2),
        "cost": round((latency / 60) * cfg["cost_per_min"], 6),
        "model": "HiSpeech",
    }
