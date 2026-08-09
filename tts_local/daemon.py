"""Local Qwen3-TTS daemon.

Resident inference server for the `tts` CLI, other applications, and agents:
- Single instance enforced with a non-blocking flock taken before anything else
- Binds 127.0.0.1 first, publishes a 0600 discovery file, then loads the model
  in a background thread; /v1/health reports loading -> ready | failed
- Async job API under /v1, bearer-token auth, bounded queue, cancellation,
  TTL cleanup of finished jobs and audio files
- Also serves the browser UI at /

Run directly (foreground): python -m tts_local.daemon
Normally started via: tts daemon start
"""

import argparse
import fcntl
import hmac
import json
import logging
import os
import queue
import re
import secrets
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from tts_local import core as tts_core
from tts_local.core import (
    CJK_RE,
    FORMATS,
    MAX_INSTRUCTION_CHARS,
    MAX_INSTRUCTION_WORDS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    SAMPLE_RATE,
    VOICES,
    Engine,
    JobCancelled,
)

from tts_local import __version__ as DAEMON_VERSION

PROTOCOL_VERSION = 1
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
QUEUE_CAPACITY = 16
MAX_JOBS = 256  # total tracked jobs (any status) before submissions get 429
JOB_TTL_SEC = 3600
CLEANUP_INTERVAL_SEC = 60
MAX_BODY_BYTES = 256 * 1024
RETRY_AFTER_SEC = 30

STATE_DIR = Path(os.environ.get("TTS_STATE_DIR", "~/.local/state/qwen-tts")).expanduser()
AUDIO_DIR = STATE_DIR / "audio"
DISCOVERY_PATH = STATE_DIR / "daemon.json"
LOCK_PATH = STATE_DIR / "daemon.lock"
LOG_PATH = STATE_DIR / "daemon.log"

INDEX_HTML = resources.files("tts_local").joinpath("index.html")

CONTENT_TYPES = {"m4a": "audio/mp4", "wav": "audio/wav"}
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; media-src 'self' blob:; img-src 'self'"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ttsd")

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# ---------------------------------------------------------------- daemon state

engine = Engine()
model_state = {"status": "loading", "error": None}  # loading | ready | failed
started_at = time.time()
token = ""

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
# Unbounded; capacity is enforced by counting queued jobs under jobs_lock so
# that canceling a queued job frees its slot immediately.
work_queue: "queue.Queue[str]" = queue.Queue()


def error_body(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": {"code": code, "message": message, "retryable": retryable}}


# ------------------------------------------------------------------- lifecycle


def acquire_lock():
    """Take the single-instance lock; returns the held fd or exits."""
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        log.error("another daemon holds %s; exiting", LOCK_PATH)
        sys.exit(1)
    return fd


def write_discovery(port: int) -> None:
    payload = {
        "pid": os.getpid(),
        "port": port,
        "token": token,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "created_at": time.time(),
    }
    tmp = DISCOVERY_PATH.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, DISCOVERY_PATH)


def remove_discovery() -> None:
    """Remove the discovery file only if it still describes this process."""
    try:
        data = json.loads(DISCOVERY_PATH.read_text())
        if data.get("pid") == os.getpid():
            DISCOVERY_PATH.unlink()
    except (OSError, ValueError):
        pass


def load_model() -> None:
    # Never silently pull 3.5 GB: require the model to be pre-downloaded
    # (tts setup) unless the caller explicitly opted in.
    if os.environ.get("TTS_AUTO_DOWNLOAD") != "1" and not tts_core.model_cached():
        model_state["status"] = "failed"
        model_state["error"] = (
            "model not downloaded; run `tts setup` once (~3.5 GB download), "
            "or set TTS_AUTO_DOWNLOAD=1"
        )
        log.error(model_state["error"])
        return
    log.info("loading %s ...", MODEL_ID)
    t0 = time.time()
    try:
        engine.load()
    except Exception as exc:
        model_state["status"] = "failed"
        model_state["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("model load failed")
        return
    model_state["status"] = "ready"
    log.info("model loaded in %.1fs (sample_rate=%d)", time.time() - t0, SAMPLE_RATE)


# ------------------------------------------------------------------- worker


def synthesize_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        # Canceled-then-expired jobs may be gone by the time they dequeue.
        if job is None or job["status"] != "queued":
            return
        job["status"] = "running"
        job["started_at"] = time.time()
    cancel_event = job["cancel_event"]

    def on_progress(done, total, meta):
        with jobs_lock:
            job["chunks_done"] = done
            job["chunks_total"] = total
            job["voice"] = meta["voice"]
            job["resolved_language"] = meta["language"]

    t0 = time.time()
    audio, meta = engine.synthesize(
        job["text"],
        language=job["language"],
        instruction=job["instruction"],
        cancel=cancel_event.is_set,
        progress=on_progress,
    )

    if cancel_event.is_set():
        raise JobCancelled()

    wav_path = AUDIO_DIR / f"{job_id}.wav"
    tts_core.write_wav(wav_path, audio)
    if job["format"] == "m4a":
        m4a_path = AUDIO_DIR / f"{job_id}.m4a"
        tts_core.encode_m4a(wav_path, m4a_path)
        wav_path.unlink()
        out_path = m4a_path
    else:
        out_path = wav_path

    if cancel_event.is_set():  # canceled during encoding
        out_path.unlink(missing_ok=True)
        raise JobCancelled()

    generation = time.time() - t0
    duration = len(audio) / SAMPLE_RATE
    log.info(
        "job %s: done — %.2fs audio in %.2fs (rtf %.2f), %s",
        job_id[:8], duration, generation,
        generation / duration if duration else 0.0, out_path.name,
    )
    with jobs_lock:
        job["status"] = "done"
        job["completed_at"] = time.time()
        job["duration_sec"] = round(duration, 2)
        job["generation_sec"] = round(generation, 2)
        job["path"] = str(out_path)


def worker_loop() -> None:
    while True:
        job_id = work_queue.get()
        try:
            synthesize_job(job_id)
        except JobCancelled:
            with jobs_lock:
                job = jobs.get(job_id)
                if job:
                    job["status"] = "canceled"
                    job["completed_at"] = time.time()
            log.info("job %s: canceled", job_id[:8])
        except Exception as exc:  # keep the worker alive on any job failure
            log.exception("job %s failed", job_id[:8])
            with jobs_lock:
                job = jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["completed_at"] = time.time()
                    job["error"] = {
                        "code": "synthesis_failed",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
        finally:
            work_queue.task_done()


def cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        try:
            cleanup_expired()
        except Exception:
            log.exception("cleanup pass failed")


def cleanup_expired() -> None:
    now = time.time()
    with jobs_lock:
        expired = [
            j for j in jobs.values()
            if j.get("completed_at") and now - j["completed_at"] > JOB_TTL_SEC
        ]
        for job in expired:
            del jobs[job["id"]]
    for job in expired:
        if job.get("path"):
            Path(job["path"]).unlink(missing_ok=True)
    # Orphans from previous daemon runs.
    for f in AUDIO_DIR.iterdir():
        if now - f.stat().st_mtime > JOB_TTL_SEC:
            f.unlink(missing_ok=True)


# ------------------------------------------------------------------- handler


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"qwen-tts/{DAEMON_VERSION}"

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    # ---- helpers

    def _send_json(self, obj: dict, status: int = 200, headers: dict | None = None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status: int, code: str, message: str, retryable: bool = False) -> None:
        # The request body may be unread; drop the connection rather than let
        # leftover bytes be parsed as the next keep-alive request.
        self.close_connection = True
        headers = {"Retry-After": str(RETRY_AFTER_SEC)} if status in (429, 503) else None
        self._send_json(error_body(code, message, retryable), status, headers)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        supplied = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        if supplied and hmac.compare_digest(supplied, token):
            return True
        self._fail(401, "unauthorized", "missing or invalid bearer token")
        return False

    def _job_snapshot(self, job_id: str) -> dict | None:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                return None
            snapshot = {
                k: job[k]
                for k in (
                    "id", "status", "format", "chunks_done", "chunks_total", "error",
                    "duration_sec", "generation_sec", "voice", "resolved_language",
                    "created_at", "started_at", "completed_at",
                )
                if k in job
            }
            if job["status"] == "queued":
                queued = sorted(
                    (j["created_at"], j["id"]) for j in jobs.values() if j["status"] == "queued"
                )
                snapshot["queue_position"] = next(
                    (i for i, (_, jid) in enumerate(queued) if jid == job_id), 0
                )
        if snapshot["status"] == "done":
            snapshot["audio_url"] = f"/v1/jobs/{job_id}/audio"
            snapshot["sample_rate"] = SAMPLE_RATE
        return snapshot

    # ---- routes

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            # No token in the page: / is unauthenticated, so an embedded token
            # would leak to any local process. The UI reads it from the URL
            # fragment instead (see `tts daemon status`).
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/health":
            with jobs_lock:
                queued = sum(1 for j in jobs.values() if j["status"] == "queued")
            self._send_json({
                "status": model_state["status"],
                "error": model_state["error"],
                "protocol_version": PROTOCOL_VERSION,
                "daemon_version": DAEMON_VERSION,
                "model_id": MODEL_ID,
                "pid": os.getpid(),
                "uptime_sec": round(time.time() - started_at, 1),
                "jobs_queued": queued,
                "queue_capacity": QUEUE_CAPACITY,
            })
            return

        if not self._authorized():
            return

        if self.path == "/v1/voices":
            self._send_json({
                "voices": [
                    {"name": name, "language": lang} for lang, name in VOICES.items()
                ],
                "languages": ["auto", *VOICES],
                "formats": list(FORMATS),
                "default_format": "m4a",
                "max_text_chars": MAX_TEXT_CHARS,
            })
            return

        m = re.fullmatch(r"/v1/jobs/([0-9a-f]{32})", self.path)
        if m:
            snapshot = self._job_snapshot(m.group(1))
            if snapshot is None:
                self._fail(404, "not_found", "unknown or expired job id")
                return
            self._send_json(snapshot)
            return

        m = re.fullmatch(r"/v1/jobs/([0-9a-f]{32})/audio", self.path)
        if m:
            with jobs_lock:
                job = jobs.get(m.group(1))
                path = job.get("path") if job and job["status"] == "done" else None
                fmt = job["format"] if job else "m4a"
            if not path:
                self._fail(404, "not_found", "audio not available")
                return
            try:
                data = Path(path).read_bytes()
            except OSError:  # TTL cleanup may have raced us
                self._fail(404, "not_found", "audio expired")
                return
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES[fmt])
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Disposition", f'inline; filename="tts-{m.group(1)[:8]}.{fmt}"'
            )
            self.end_headers()
            self.wfile.write(data)
            return

        self._fail(404, "not_found", "not found")

    def do_POST(self):
        if not self._authorized():
            return

        if self.path == "/v1/shutdown":
            self._send_json({"ok": True})
            log.info("shutdown requested via API")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if self.path != "/v1/jobs":
            self._fail(404, "not_found", "not found")
            return

        if model_state["status"] == "loading":
            self._fail(503, "loading", "model is still loading", retryable=True)
            return
        if model_state["status"] == "failed":
            self._fail(500, "model_failed", model_state["error"] or "model failed to load")
            return

        payload = self._read_json_body()
        if payload is None:
            return

        text = payload.get("text")
        language = payload.get("language", "auto")
        # No default instruction: style instructions can trigger runaway
        # generation (model runs to its length cap) on short/Chinese text.
        instruction = payload.get("instruction", "")
        fmt = payload.get("format", "m4a")
        if not isinstance(text, str) or not text.strip():
            self._fail(400, "invalid_request", "text (non-empty string) is required")
            return
        text = text.strip()
        if len(text) > MAX_TEXT_CHARS:
            self._fail(400, "invalid_request", f"text exceeds {MAX_TEXT_CHARS} character cap")
            return
        if not isinstance(language, str) or language.strip().lower() not in ("auto", *VOICES):
            self._fail(400, "invalid_request", "language must be auto, english, or chinese")
            return
        if not isinstance(instruction, str) or len(instruction) > MAX_INSTRUCTION_CHARS:
            self._fail(
                400, "invalid_request",
                f"instruction must be a string of at most {MAX_INSTRUCTION_CHARS} chars",
            )
            return
        instruction = instruction.strip()
        contains_cjk = CJK_RE.search(instruction) is not None
        if contains_cjk:
            if len(instruction) > 36:
                self._fail(
                    400, "invalid_request",
                    f"instruction too long ({len(instruction)} chars, max 36 for Chinese). "
                    "This model follows short style cues best — emotion, pace, one or two "
                    "vocal qualities.",
                )
                return
        else:
            word_count = len(instruction.split())
            if word_count > MAX_INSTRUCTION_WORDS:
                self._fail(
                    400, "invalid_request",
                    f"instruction too long ({word_count} words, max {MAX_INSTRUCTION_WORDS}). "
                    "This model follows short style cues best — emotion, pace, one or two "
                    'vocal qualities. Example: "Speak in a sad, low tone, voice heavy and slow."',
                )
                return
        if fmt not in FORMATS:
            self._fail(400, "invalid_request", f"format must be one of {', '.join(FORMATS)}")
            return

        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "status": "queued",
            "text": text,
            "language": language.strip().lower(),
            "instruction": instruction,
            "format": fmt,
            "chunks_done": 0,
            "created_at": time.time(),
            "cancel_event": threading.Event(),
        }
        with jobs_lock:
            queued = sum(1 for j in jobs.values() if j["status"] == "queued")
            # len(jobs) cap bounds canceled-but-undequeued tombstones too;
            # TTL cleanup keeps it from pinning at the cap.
            if queued >= QUEUE_CAPACITY or len(jobs) >= MAX_JOBS:
                full = True
            else:
                full = False
                jobs[job_id] = job
        if full:
            self._fail(429, "busy", "job queue is full, retry later", retryable=True)
            return
        work_queue.put(job_id)
        log.info("job %s: queued (%d chars, language=%s, format=%s)",
                 job_id[:8], len(text), job["language"], fmt)
        self._send_json({"id": job_id, "status": "queued"}, 202)

    def do_DELETE(self):
        if not self._authorized():
            return
        m = re.fullmatch(r"/v1/jobs/([0-9a-f]{32})", self.path)
        if not m:
            self._fail(404, "not_found", "not found")
            return
        job_id = m.group(1)
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                self._fail(404, "not_found", "unknown or expired job id")
                return
            if job["status"] == "queued":
                # Stays in the queue; the worker skips jobs already canceled.
                job["status"] = "canceled"
                job["completed_at"] = time.time()
            elif job["status"] == "running":
                job["cancel_event"].set()  # observed between chunks
        self._send_json(self._job_snapshot(job_id))

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._fail(413, "invalid_request", "request body missing or too large")
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._fail(400, "invalid_request", "invalid JSON")
            return None
        if not isinstance(payload, dict):
            self._fail(400, "invalid_request", "body must be a JSON object")
            return None
        return payload


# ---------------------------------------------------------------------- main


def bind_server(preferred_port: int) -> ThreadingHTTPServer:
    try:
        return ThreadingHTTPServer((HOST, preferred_port), Handler)
    except OSError:
        log.warning("port %d unavailable, falling back to an ephemeral port", preferred_port)
        return ThreadingHTTPServer((HOST, 0), Handler)


def main() -> None:
    global token

    parser = argparse.ArgumentParser(description="Local Qwen3-TTS daemon")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"preferred port (default {DEFAULT_PORT}; falls back to ephemeral)")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    AUDIO_DIR.mkdir(exist_ok=True)

    lock_fd = acquire_lock()  # held for process lifetime  # noqa: F841

    token = secrets.token_urlsafe(32)
    server = bind_server(args.port)
    port = server.server_address[1]
    write_discovery(port)
    log.info("listening on http://%s:%d (pid %d)", HOST, port, os.getpid())

    threading.Thread(target=load_model, daemon=True, name="model-loader").start()
    threading.Thread(target=worker_loop, daemon=True, name="tts-worker").start()
    threading.Thread(target=cleanup_loop, daemon=True, name="tts-cleanup").start()

    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(
        target=server.shutdown, daemon=True).start())

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("shutting down")
        server.server_close()
        remove_discovery()


if __name__ == "__main__":
    main()
