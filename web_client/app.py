#!/usr/bin/env python3
"""Web UI for Qwen3-VL Video Context API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_BASE = os.environ.get("VLM_API_URL", "http://127.0.0.1:8080")
UPLOAD_MAX_MB = int(os.environ.get("VLM_UPLOAD_MAX_MB", "256"))
DEBUG = os.environ.get("WEB_CLIENT_DEBUG", "1") != "0"


def fetch_models() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/v1/models", timeout=3)
        if r.ok:
            return r.json().get("data", [])
    except requests.RequestException:
        pass
    return []


def model_label(model_id: str) -> str:
    lower = model_id.lower()
    if "0.8" in lower:
        return f"{model_id} (быстрее)"
    if "2b" in lower:
        return f"{model_id} (точнее)"
    return model_id


@app.route("/", methods=["GET"])
def index():
    status = {}
    try:
        r = requests.get(f"{API_BASE}/v1/status", timeout=3)
        if r.ok:
            status = r.json()
    except requests.RequestException:
        status = {"status": "unreachable", "error": f"Cannot reach {API_BASE}"}

    models = fetch_models()
    return render_template(
        "index.html",
        status=status,
        api_base=API_BASE,
        models=models,
        model_label=model_label,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return render_template(
            "result.html",
            ok=False,
            error="Выберите видеофайл",
            api_base=API_BASE,
        )

    video = request.files["video"]
    if not video.filename:
        return render_template(
            "result.html",
            ok=False,
            error="Пустое имя файла",
            api_base=API_BASE,
        )

    frames = request.form.get("frames", "8")
    frame_budget = request.form.get("frame_budget", "")
    max_tokens = request.form.get("max_tokens", "")
    enable_thinking = request.form.get("enable_thinking", "")
    temperature = request.form.get("temperature", "")
    lang = request.form.get("lang", "ru")
    prompt_mode = request.form.get("prompt_mode", "detailed")
    transcript = request.form.get("transcript", "")
    model = request.form.get("model", "")

    data = {
        "model": model,
        "frames": frames,
        "lang": lang,
        "prompt_mode": prompt_mode,
    }
    if frame_budget.strip():
        data["frame_budget"] = frame_budget.strip()
    if max_tokens.strip():
        data["max_tokens"] = max_tokens.strip()
    if enable_thinking.strip():
        data["enable_thinking"] = enable_thinking.strip()
    if temperature.strip():
        data["temperature"] = temperature.strip()
    if transcript.strip():
        data["transcript"] = transcript.strip()

    try:
        video.stream.seek(0, os.SEEK_END)
        size_mb = video.stream.tell() / (1024 * 1024)
        video.stream.seek(0)
        if size_mb <= 0:
            raise ValueError("Пустой видеофайл")
        if size_mb > UPLOAD_MAX_MB:
            raise ValueError(f"Файл слишком большой ({size_mb:.1f} MB, лимит {UPLOAD_MAX_MB} MB)")

        resp = requests.post(
            f"{API_BASE}/v1/video/analyze",
            files={"file": (video.filename, video.stream, video.mimetype or "video/mp4")},
            data=data,
            timeout=3600,
        )
    except requests.RequestException as exc:
        return render_template(
            "result.html",
            ok=False,
            error=f"Ошибка API: {exc}",
            api_base=API_BASE,
        )
    except ValueError as exc:
        return render_template(
            "result.html",
            ok=False,
            error=str(exc),
            api_base=API_BASE,
        )

    if not resp.ok:
        return render_template(
            "result.html",
            ok=False,
            error=f"API {resp.status_code}: {resp.text[:500]}",
            api_base=API_BASE,
        )

    payload = resp.json()
    return render_template(
        "result.html",
        ok=payload.get("error") in (None, ""),
        result=payload,
        result_json=json.dumps(payload, ensure_ascii=False, indent=2),
        api_base=API_BASE,
    )


if __name__ == "__main__":
    host = os.environ.get("WEB_CLIENT_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_CLIENT_PORT", "5000"))
    app.run(host=host, port=port, debug=DEBUG, use_reloader=DEBUG)
