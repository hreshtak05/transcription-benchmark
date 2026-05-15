import os
import time
import httpx


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    token = os.getenv("WAVAM_API_KEY", "")
    project_id = os.getenv("WAVAM_PROJECT_ID", "")

    if not token:
        raise ValueError("WAVAM_API_KEY is not set in .env")
    if not project_id:
        raise ValueError("WAVAM_PROJECT_ID is not set in .env — get it from wav.am dashboard")

    cost_per_min = float(os.getenv("WAVAM_COST_PER_MINUTE", "0"))
    start = time.time()

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            "https://wav.am/transcribe_audio/",
            headers={"Authorization": token},
            data={
                "project_id": project_id,
                "language": "hy",
                "num_speakers": "2",
            },
            files={"audio_file": (filename, audio_bytes, "audio/mpeg")},
        )
        response.raise_for_status()
        data = response.json()

    latency = time.time() - start

    if isinstance(data, list):
        data = data[0] if data else {}

    return {
        "text": data.get("text", "").strip(),
        "latency": round(latency, 2),
        "cost": round((latency / 60) * cost_per_min, 6),
        "model": "wav.am",
    }
