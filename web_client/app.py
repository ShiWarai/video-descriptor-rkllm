#!/usr/bin/env python3
"""Web UI for video-descriptor-rkllm API."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, make_response, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

API_BASE = os.environ.get("VLM_API_URL", "http://127.0.0.1:8080")
API_KEY = os.environ.get("VLM_API_KEY", "")
UPLOAD_MAX_MB = int(os.environ.get("VLM_UPLOAD_MAX_MB", "256"))
DEBUG = os.environ.get("WEB_CLIENT_DEBUG", "0") != "0"
UPLOAD_DIR = Path(os.environ.get("WEB_UPLOAD_DIR", tempfile.gettempdir())) / "vlm_web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AnalyzeJobState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    LOST = "lost"


@dataclass
class AnalyzeJob:
    state: AnalyzeJobState = AnalyzeJobState.ACCEPTED
    error: str = ""
    result: dict[str, Any] | None = None
    client_metrics: dict[str, float] = field(default_factory=dict)
    video_path: Path | None = None


_jobs: dict[str, AnalyzeJob] = {}
_jobs_lock = threading.Lock()
_api_slot_lock = threading.Lock()


def api_headers() -> dict[str, str]:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def fetch_models() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/v1/models", headers=api_headers(), timeout=3)
        if r.ok:
            return r.json().get("data", [])
    except requests.RequestException:
        pass
    return []


def analyze_response(*, ok: bool, submit_status: str, **template_kwargs):
    html = render_template("result.html", ok=ok, **template_kwargs)
    resp = make_response(html)
    resp.headers["X-VLM-Submit-Status"] = submit_status
    return resp


def model_label(model_id: str) -> str:
    lower = model_id.lower()
    if "0.8" in lower:
        return f"{model_id} (быстрее)"
    if "4b" in lower:
        return f"{model_id} (максимальная точность)"
    if "2b" in lower:
        return f"{model_id} (точнее)"
    return model_id


def _cleanup_video(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _parse_analyze_form() -> tuple[dict[str, str], str | None]:
    data = {
        "model": request.form.get("model", ""),
        "frames": request.form.get("frames", "16"),
        "lang": request.form.get("lang", "ru"),
        "prompt_mode": request.form.get("prompt_mode", "detailed"),
    }
    for key in ("frame_budget", "max_tokens", "enable_thinking", "temperature"):
        value = request.form.get(key, "").strip()
        if value:
            data[key] = value
    return data, request.form.get("submit_token", "").strip() or None


def _save_uploaded_video(video) -> Path:
    video.stream.seek(0, os.SEEK_END)
    size_mb = video.stream.tell() / (1024 * 1024)
    video.stream.seek(0)
    if size_mb <= 0:
        raise ValueError("Пустой видеофайл")
    if size_mb > UPLOAD_MAX_MB:
        raise ValueError(f"Файл слишком большой ({size_mb:.1f} MB, лимит {UPLOAD_MAX_MB} MB)")

    safe_name = secure_filename(video.filename) or "upload.bin"
    path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_name}"
    video.save(path)
    return path


def _run_api_analyze(token: str, video_path: Path, data: dict[str, str], mimetype: str, filename: str) -> None:
    with _jobs_lock:
        job = _jobs.get(token)
        if job is None:
            return
        job.state = AnalyzeJobState.RUNNING

    try:
        with video_path.open("rb") as stream:
            t_client = time.perf_counter()
            resp = requests.post(
                f"{API_BASE}/v1/video/analyze",
                files={"file": (filename, stream, mimetype or "application/octet-stream")},
                data=data,
                headers=api_headers(),
                timeout=3600,
            )
            client_wall_ms = (time.perf_counter() - t_client) * 1000.0
    except (requests.ConnectionError, requests.Timeout) as exc:
        with _jobs_lock:
            job = _jobs.get(token)
            if job is None:
                return
            job.state = AnalyzeJobState.LOST
            job.error = (
                f"Связь с API потеряна: {exc}. "
                "Задача могла остаться в обработке на API — повтор не выполняется автоматически."
            )
        return
    except requests.RequestException as exc:
        with _jobs_lock:
            job = _jobs.get(token)
            if job is None:
                return
            job.state = AnalyzeJobState.FAILED
            job.error = f"Ошибка API: {exc}"
        return
    finally:
        _cleanup_video(video_path)
        _api_slot_lock.release()

    with _jobs_lock:
        job = _jobs.get(token)
        if job is None:
            return
        job.client_metrics["client_wall_ms"] = client_wall_ms

    if not resp.ok:
        err_text = resp.text[:500]
        try:
            err_json = resp.json()
            if isinstance(err_json, dict) and err_json.get("error"):
                err_text = str(err_json["error"])
        except Exception:
            pass
        with _jobs_lock:
            job = _jobs.get(token)
            if job is None:
                return
            job.state = AnalyzeJobState.FAILED
            job.error = err_text
        return

    payload = resp.json()
    with _jobs_lock:
        job = _jobs.get(token)
        if job is None:
            return
        job.result = payload
        if payload.get("error") in (None, ""):
            job.state = AnalyzeJobState.DONE
        else:
            job.state = AnalyzeJobState.FAILED
            job.error = str(payload.get("error"))


def _start_analyze_job(token: str) -> tuple[AnalyzeJob | None, str | None, int]:
    """Returns (job, error_message, http_status). Idempotent per submit_token."""
    with _jobs_lock:
        existing = _jobs.get(token)
        if existing is not None:
            return existing, None, 200

        if not _api_slot_lock.acquire(blocking=False):
            return None, "Уже выполняется другая задача. Повторная отправка отклонена.", 409

        job = AnalyzeJob()
        _jobs[token] = job

    try:
        if "video" not in request.files:
            with _jobs_lock:
                _jobs.pop(token, None)
            _api_slot_lock.release()
            return None, "Выберите видео или GIF", 400

        video = request.files["video"]
        if not video.filename:
            with _jobs_lock:
                _jobs.pop(token, None)
            _api_slot_lock.release()
            return None, "Пустое имя файла", 400

        data, _ = _parse_analyze_form()
        try:
            video_path = _save_uploaded_video(video)
        except ValueError as exc:
            with _jobs_lock:
                _jobs.pop(token, None)
            _api_slot_lock.release()
            return None, str(exc), 400

        with _jobs_lock:
            job = _jobs[token]
            job.video_path = video_path

        thread = threading.Thread(
            target=_run_api_analyze,
            args=(token, video_path, data, video.mimetype or "", video.filename),
            daemon=True,
            name=f"analyze-{token[:8]}",
        )
        thread.start()
        return job, None, 202
    except Exception:
        with _jobs_lock:
            _jobs.pop(token, None)
        _api_slot_lock.release()
        raise


@app.route("/api/jobs/<job_id>", methods=["GET"])
def api_job_progress(job_id: str):
    try:
        r = requests.get(
            f"{API_BASE}/v1/jobs/{job_id}",
            headers=api_headers(),
            timeout=3,
        )
        if r.ok:
            return r.json(), 200
        try:
            body = r.json()
        except ValueError:
            body = {"error": r.text[:500]}
        return body, r.status_code
    except requests.RequestException as exc:
        return {"error": str(exc)}, 502


@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        r = requests.get(f"{API_BASE}/v1/status", headers=api_headers(), timeout=3)
        if r.ok:
            return r.json(), 200
        return {"status": "error", "error": f"API {r.status_code}"}, 502
    except requests.RequestException as exc:
        return {"status": "unreachable", "error": str(exc)}, 502


@app.route("/api/analyze", methods=["POST"])
def api_analyze_start():
    _, token = _parse_analyze_form()
    if not token:
        return jsonify({"error": "submit_token required"}), 400

    job, error, status = _start_analyze_job(token)
    if error:
        return jsonify({"error": error, "submit_token": token}), status

    assert job is not None
    return (
        jsonify(
            {
                "submit_token": token,
                "status": job.state.value,
            }
        ),
        status,
    )


@app.route("/api/analyze/<token>", methods=["GET"])
def api_analyze_poll(token: str):
    with _jobs_lock:
        job = _jobs.get(token)
    if job is None:
        return jsonify({"error": "unknown submit_token", "status": "not_found"}), 404

    body: dict[str, Any] = {
        "submit_token": token,
        "status": job.state.value,
    }
    if job.error:
        body["error"] = job.error
    if job.state == AnalyzeJobState.DONE and job.result is not None:
        body["result"] = job.result
    return jsonify(body), 200


@app.route("/result/<token>", methods=["GET"])
def analyze_result_page(token: str):
    with _jobs_lock:
        job = _jobs.get(token)
    if job is None:
        return redirect(url_for("index"))

    if job.state == AnalyzeJobState.DONE and job.result is not None:
        payload = job.result
        return render_template(
            "result.html",
            ok=payload.get("error") in (None, ""),
            result=payload,
            result_json=json.dumps(payload, ensure_ascii=False, indent=2),
            client_metrics=job.client_metrics,
            api_base=API_BASE,
        )

    if job.state in (AnalyzeJobState.FAILED, AnalyzeJobState.LOST):
        return render_template(
            "result.html",
            ok=False,
            error=job.error or "Ошибка обработки",
            api_base=API_BASE,
        )

    return redirect(url_for("index", resume=token))


@app.route("/", methods=["GET"])
def index():
    status = {}
    try:
        r = requests.get(f"{API_BASE}/v1/status", headers=api_headers(), timeout=3)
        if r.ok:
            status = r.json()
    except requests.RequestException:
        status = {"status": "unreachable", "error": f"Cannot reach {API_BASE}"}

    models = fetch_models()
    resume_token = request.args.get("resume", "").strip() or None
    return render_template(
        "index.html",
        status=status,
        api_base=API_BASE,
        models=models,
        model_label=model_label,
        resume_token=resume_token,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """Legacy HTML form POST — redirects into async flow when submit_token is present."""
    _, token = _parse_analyze_form()
    if not token:
        return analyze_response(
            ok=False,
            submit_status="failed",
            error="submit_token required",
            api_base=API_BASE,
        )

    job, error, status = _start_analyze_job(token)
    if error:
        submit_status = "failed"
        if status == 409:
            submit_status = "busy"
        return analyze_response(ok=False, submit_status=submit_status, error=error, api_base=API_BASE)

    return redirect(url_for("index", resume=token))


if __name__ == "__main__":
    host = os.environ.get("WEB_CLIENT_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_CLIENT_PORT", "5000"))
    app.run(host=host, port=port, debug=DEBUG, use_reloader=DEBUG, threaded=True)
