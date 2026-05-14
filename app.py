import json
import os
import asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from services.gemini import transcribe as gemini_transcribe
from services.wavam import transcribe as wavam_transcribe
from services.hispeech import transcribe as hispeech_transcribe
from comparison import compare, llm_judge

load_dotenv()

app = FastAPI(title="Transcription Benchmark")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {
    "gemini": gemini_transcribe,
    "wavam": wavam_transcribe,
    "hispeech": hispeech_transcribe,
}


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend", "index.html"))


@app.post("/api/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    reference: str = Form(...),
    models: str = Form(...),
):
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.mp3"
    selected = [m.strip() for m in models.split(",") if m.strip() in MODELS]

    async def stream():
        results = {}

        async def run_model(name: str):
            try:
                result = await MODELS[name](audio_bytes, filename)
                result["comparison"] = compare(reference, result["text"])
                results[name] = result
                return {"type": "result", "model": name, "data": result}
            except Exception as e:
                return {"type": "error", "model": name, "message": str(e)}

        # Run all selected models concurrently
        yield {"data": json.dumps({"type": "progress", "message": f"Running {len(selected)} model(s) in parallel..."})}

        tasks = [asyncio.create_task(run_model(name)) for name in selected]
        for coro in asyncio.as_completed(tasks):
            event = await coro
            yield {"data": json.dumps(event)}

        # Qualitative LLM analysis after all models finish
        if results:
            yield {"data": json.dumps({"type": "progress", "message": "Running qualitative analysis with Gemini..."})}
            try:
                analysis = await llm_judge(reference, results)
                yield {"data": json.dumps({"type": "analysis", "data": analysis})}
            except Exception as e:
                yield {"data": json.dumps({"type": "error", "model": "judge", "message": str(e)})}

        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(stream(), ping=15)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
