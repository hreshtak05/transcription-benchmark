import re
import os
import json
import difflib
import asyncio
import google.generativeai as genai
from jiwer import wer, cer


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def word_diff_html(reference: str, hypothesis: str) -> str:
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    parts = []

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            for w in ref_words[a0:a1]:
                parts.append(f'<span class="w-ok">{w}</span>')
        elif opcode == "replace":
            hyp_chunk = " ".join(hyp_words[b0:b1])
            for w in ref_words[a0:a1]:
                parts.append(f'<span class="w-sub" title="model said: {hyp_chunk}">{w}</span>')
            for w in hyp_words[b0:b1]:
                parts.append(f'<span class="w-ins">[{w}]</span>')
        elif opcode == "delete":
            for w in ref_words[a0:a1]:
                parts.append(f'<span class="w-del">{w}</span>')
        elif opcode == "insert":
            for w in hyp_words[b0:b1]:
                parts.append(f'<span class="w-ins">[{w}]</span>')

    return " ".join(parts)


def compare(reference: str, hypothesis: str) -> dict:
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)

    word_error = wer(ref_n, hyp_n)
    char_error = cer(ref_n, hyp_n)
    accuracy = max(0.0, 1.0 - word_error)

    ref_words = ref_n.split()
    word_count = len(ref_words)

    return {
        "wer": round(word_error * 100, 2),
        "cer": round(char_error * 100, 2),
        "accuracy": round(accuracy * 100, 2),
        "word_count": word_count,
        "diff_html": word_diff_html(reference, hypothesis),
    }


async def llm_judge(reference: str, results: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set — qualitative analysis unavailable"}

    genai.configure(api_key=api_key)

    models_block = "\n\n".join(
        f"### {name}\n{data['text']}" for name, data in results.items()
    )

    prompt = f"""You are evaluating AI transcription models on audio that may contain Armenian, Russian, and English speech.

Reference (human-verified) transcription:
{reference}

AI model outputs:
{models_block}

For EACH model, evaluate on these dimensions and return a JSON object:
{{
  "model_name": {{
    "armenian_quality": "Short assessment of Armenian script accuracy, grammar, and naturalness",
    "error_types": "What kinds of errors appear? (substitutions, omissions, hallucinations, dialect issues, script errors)",
    "multilingual_handling": "How well does it handle Armenian/Russian/English code-switching?",
    "noise_sensitivity": "Any signs the model struggles with background noise or audio quality?",
    "summary": "2-sentence overall verdict"
  }}
}}

Return ONLY valid JSON, no markdown, no extra text."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = await model.generate_content_async(prompt)
    text = response.text.strip()

    # Strip possible markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
