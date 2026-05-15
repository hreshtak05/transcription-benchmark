import os
import time
import httpx

BASE_URL = "https://wav.am"


async def _get_or_create_project(token: str) -> str:
    """Return existing benchmark project ID or create one."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{BASE_URL}/get_projects/",
            headers={"Authorization": token},
            json={},
        )
        r.raise_for_status()
        projects = r.json().get("projects", [])
        for p in projects:
            if p.get("name") == "transcription-benchmark":
                return str(p["id"])

        # Create it
        r2 = await client.post(
            f"{BASE_URL}/add_project/",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={"name": "transcription-benchmark"},
        )
        r2.raise_for_status()
        return str(r2.json()["project_id"])


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    token = os.getenv("WAVAM_API_KEY", "")
    if not token:
        raise ValueError("WAVAM_API_KEY is not set in .env")

    cost_per_min = float(os.getenv("WAVAM_COST_PER_MINUTE", "0"))
    start = time.time()

    project_id = os.getenv("WAVAM_PROJECT_ID", "")
    if not project_id:
        project_id = await _get_or_create_project(token)

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            f"{BASE_URL}/transcribe_audio/",
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
    text = data.get("text", "")

    return {
        "text": text.strip(),
        "latency": round(latency, 2),
        "cost": round((latency / 60) * cost_per_min, 6),
        "model": "wav.am",
    }
