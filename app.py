"""Local Qwen3-TTS web app.

Single-model, single-worker text-to-speech server:
- Model: mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 (loaded once at startup)
- Voices: Aiden (English), Serena (Chinese)
- Stdlib ThreadingHTTPServer bound to 127.0.0.1; one worker thread serializes inference
- Routes: POST /speak, GET /status/<id>, GET /audio/<id>.wav, GET /
"""

import json
import logging
import queue
import re
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HOST = "127.0.0.1"
PORT = 8765
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 10_000
CHUNK_TARGET = 300  # soft minimum before we close a chunk
CHUNK_MAX = 400  # hard maximum chunk length
PARAGRAPH_PAUSE_MS = 250
CHUNK_PAUSE_MS = 80
MAX_BODY_BYTES = 256 * 1024
DEFAULT_INSTRUCTION = (
    "Calm, clear audiobook narration. Natural pacing, restrained emotion, "
    "precise pronunciation, and comfortable pauses between paragraphs."
)

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
INDEX_HTML = APP_DIR / "index.html"

VOICES = {"english": "Aiden", "chinese": "Serena"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tts")

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work_queue: "queue.Queue[str]" = queue.Queue()

model = None  # set once in main()


# ---------------------------------------------------------------- text utils

CJK_RE = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

# Sentence boundary: split after terminal punctuation (+ optional closing quotes).
SENTENCE_RE = re.compile(r"[^.!?。！？；;…\n]*[.!?。！？；;…]+[\"'”’）)\]]*|[^.!?。！？；;…\n]+")


def detect_language(text: str) -> str:
    """Return 'chinese' or 'english' based on script mix."""
    cjk = len(CJK_RE.findall(text))
    ascii_letters = len(ASCII_LETTER_RE.findall(text))
    if cjk == 0:
        return "english"
    if ascii_letters == 0:
        return "chinese"
    return "chinese" if cjk / (cjk + ascii_letters) >= 0.3 else "english"


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(paragraph) if s.strip()]


def hard_split(sentence: str, limit: int) -> list[str]:
    """Split an over-long sentence at commas/spaces, else at a hard boundary."""
    parts = []
    rest = sentence
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind(c) for c in ("，", ","))
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit - 1
        parts.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].strip()
    if rest:
        parts.append(rest)
    return parts


def chunk_text(text: str) -> list[list[str]]:
    """Return paragraphs, each a list of 300-400 char chunks of whole sentences."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result = []
    for para in paragraphs:
        para = re.sub(r"\s*\n\s*", " ", para)
        pieces: list[str] = []
        for sentence in split_sentences(para):
            if len(sentence) > CHUNK_MAX:
                pieces.extend(hard_split(sentence, CHUNK_MAX))
            else:
                pieces.append(sentence)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if current and (len(candidate) > CHUNK_MAX or len(current) >= CHUNK_TARGET):
                chunks.append(current)
                current = piece
            else:
                current = candidate
        if current:
            chunks.append(current)
        if chunks:
            result.append(chunks)
    return result


# ------------------------------------------------------------------- worker


def synthesize_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job["status"] = "running"
        job["started_at"] = time.time()

    language = job["language"]
    if language == "auto":
        language = detect_language(job["text"])
    speaker = VOICES[language]

    paragraphs = chunk_text(job["text"])
    total_chunks = sum(len(p) for p in paragraphs)
    with jobs_lock:
        job["chunks_total"] = total_chunks
        job["voice"] = speaker
        job["resolved_language"] = language
    log.info(
        "job %s: %d paragraph(s), %d chunk(s), voice=%s lang=%s",
        job_id[:8], len(paragraphs), total_chunks, speaker, language,
    )

    para_pause = np.zeros(int(SAMPLE_RATE * PARAGRAPH_PAUSE_MS / 1000), dtype=np.float32)
    chunk_pause = np.zeros(int(SAMPLE_RATE * CHUNK_PAUSE_MS / 1000), dtype=np.float32)

    segments: list[np.ndarray] = []
    done = 0
    t_job = time.time()
    for p_idx, chunks in enumerate(paragraphs):
        if p_idx > 0:
            segments.append(para_pause)
        for c_idx, chunk in enumerate(chunks):
            if c_idx > 0:
                segments.append(chunk_pause)
            t_chunk = time.time()
            audio_parts = [
                np.asarray(r.audio, dtype=np.float32)
                for r in model.generate_custom_voice(
                    text=chunk,
                    speaker=speaker,
                    language=language,
                    instruct=job["instruction"] or None,
                )
            ]
            audio = np.concatenate(audio_parts)
            dt = time.time() - t_chunk
            dur = len(audio) / SAMPLE_RATE
            log.info(
                "job %s: chunk %d/%d (%d chars) -> %.2fs audio in %.2fs (rtf %.2f)",
                job_id[:8], done + 1, total_chunks, len(chunk), dur, dt,
                dt / dur if dur else 0.0,
            )
            segments.append(audio)
            done += 1
            with jobs_lock:
                job["chunks_done"] = done

    full = np.concatenate(segments)
    pcm = np.clip(full, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)

    out_path = OUTPUT_DIR / f"{job_id}.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    total_dt = time.time() - t_job
    duration = len(full) / SAMPLE_RATE
    log.info(
        "job %s: done — %.2fs audio in %.2fs (rtf %.2f), %s",
        job_id[:8], duration, total_dt, total_dt / duration if duration else 0.0,
        out_path.name,
    )
    with jobs_lock:
        job["status"] = "done"
        job["duration_sec"] = round(duration, 2)
        job["generation_sec"] = round(total_dt, 2)
        job["path"] = str(out_path)


def worker_loop() -> None:
    while True:
        job_id = work_queue.get()
        try:
            synthesize_job(job_id)
        except Exception as exc:  # keep the worker alive on any job failure
            log.exception("job %s failed", job_id[:8])
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            work_queue.task_done()


# ------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # route access logs through logging
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        m = re.fullmatch(r"/status/([0-9a-f]{32})", self.path)
        if m:
            with jobs_lock:
                job = jobs.get(m.group(1))
                snapshot = None if job is None else {
                    k: job[k]
                    for k in (
                        "id", "status", "chunks_done", "chunks_total", "error",
                        "duration_sec", "generation_sec", "voice", "resolved_language",
                    )
                    if k in job
                }
            if snapshot is None:
                self._send_error_json(404, "unknown job id")
                return
            if snapshot["status"] == "queued":
                snapshot["queue_position"] = self._queue_position(snapshot["id"])
            if snapshot["status"] == "done":
                snapshot["audio_url"] = f"/audio/{snapshot['id']}.wav"
            self._send_json(snapshot)
            return

        m = re.fullmatch(r"/audio/([0-9a-f]{32})\.wav", self.path)
        if m:
            with jobs_lock:
                job = jobs.get(m.group(1))
                path = job.get("path") if job and job["status"] == "done" else None
            if not path:
                self._send_error_json(404, "audio not available")
                return
            data = Path(path).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Disposition", f'inline; filename="tts-{m.group(1)[:8]}.wav"'
            )
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_error_json(404, "not found")

    @staticmethod
    def _queue_position(job_id: str) -> int:
        with jobs_lock:
            queued = sorted(
                (j["created_at"], j["id"]) for j in jobs.values() if j["status"] == "queued"
            )
        return next((i for i, (_, jid) in enumerate(queued) if jid == job_id), 0)

    def do_POST(self):
        if self.path != "/speak":
            self._send_error_json(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_error_json(413, "request body missing or too large")
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error_json(400, "invalid JSON")
            return

        text_value = payload.get("text")
        language_value = payload.get("language", "auto")
        instruction_value = payload.get("instruction", DEFAULT_INSTRUCTION)
        if not isinstance(text_value, str):
            self._send_error_json(400, "text must be a string")
            return
        if not isinstance(language_value, str):
            self._send_error_json(400, "language must be a string")
            return
        if not isinstance(instruction_value, str):
            self._send_error_json(400, "instruction must be a string")
            return

        text = text_value.strip()
        language = language_value.strip().lower()
        instruction = instruction_value.strip()

        if not text:
            self._send_error_json(400, "text is required")
            return
        if len(text) > MAX_TEXT_CHARS:
            self._send_error_json(400, f"text exceeds {MAX_TEXT_CHARS} character cap")
            return
        if language not in ("auto", "english", "chinese"):
            self._send_error_json(400, "language must be auto, english, or chinese")
            return
        if len(instruction) > 500:
            self._send_error_json(400, "instruction too long (500 char max)")
            return

        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "text": text,
                "language": language,
                "instruction": instruction,
                "chunks_done": 0,
                "created_at": time.time(),
            }
        work_queue.put(job_id)
        log.info("job %s: queued (%d chars, language=%s)", job_id[:8], len(text), language)
        self._send_json({"id": job_id, "status": "queued"}, 202)


def main() -> None:
    global model
    OUTPUT_DIR.mkdir(exist_ok=True)

    from mlx_audio.tts.utils import load

    log.info("loading %s ...", MODEL_ID)
    t0 = time.time()
    model = load(MODEL_ID)
    if model.sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"expected native {SAMPLE_RATE} Hz output, model reports {model.sample_rate} Hz"
        )
    log.info("model loaded in %.1fs (sample_rate=%d)", time.time() - t0, model.sample_rate)

    threading.Thread(target=worker_loop, daemon=True, name="tts-worker").start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("serving on http://%s:%d", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
