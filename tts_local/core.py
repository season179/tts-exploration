"""Synthesis engine for the local Qwen3-TTS daemon.

Pure inference layer: model loading, text chunking, synthesis, and audio
encoding. No HTTP or job-queue concerns.
"""

import logging
import subprocess
import wave
from pathlib import Path

import numpy as np
import re

log = logging.getLogger("tts_core")

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
# Pinned so upstream pushes can't trigger surprise multi-GB redownloads or
# silently change model behavior.
MODEL_REVISION = "52f4770fd9726457eae3d3b6aa92047a25a10776"
SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 10_000
MAX_INSTRUCTION_CHARS = 500
CHUNK_TARGET = 300  # soft minimum before we close a chunk
CHUNK_MAX = 400  # hard maximum chunk length
PARAGRAPH_PAUSE_MS = 250
CHUNK_PAUSE_MS = 80
M4A_BITRATE = 64000  # AAC, plenty for 24 kHz mono speech

# Runaway-generation guard. The model sometimes fails to emit end-of-speech
# (seen with style instructions on short text) and runs to its 4096-token
# (~328s) ceiling. Cap each chunk's audio at a generous multiple of its text
# length so a runaway wastes seconds, not minutes. Normal speech stays far
# below 0.6 s/char (slow Chinese ~0.5, English ~0.1).
CODEC_TOKENS_PER_SEC = 12.5  # 4096 tokens == 327.68 s
CAP_SEC_PER_CHAR = 0.6
CAP_SEC_FLOOR = 10.0

VOICES = {"english": "Aiden", "chinese": "Serena"}
FORMATS = ("m4a", "wav")

# CJK punctuation, unified ideographs (+ext A), compatibility ideographs,
# and half/fullwidth forms. Kept as \u escapes: some endpoints have homoglyphs.
CJK_RE = re.compile("[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

# Sentence boundary: split after terminal punctuation (+ optional closing quotes).
SENTENCE_RE = re.compile(r"[^.!?。！？；;…\n]*[.!?。！？；;…]+[\"'”’）)\]]*|[^.!?。！？；;…\n]+")


class JobCancelled(Exception):
    """Raised inside synthesize() when the cancel callback returns True."""


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


def cached_model_path() -> str | None:
    """Local snapshot path if the pinned model revision is fully cached, else None.

    Any cache problem (missing, corrupt, permissions) reports as "not cached";
    `tts setup` re-verifies and repairs via download_model().
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except Exception:
        return None


def model_cached() -> bool:
    return cached_model_path() is not None


def download_model() -> str:
    """Fetch (or verify) the pinned model revision; returns its snapshot path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION)


class Engine:
    """Owns the loaded model. Not thread-safe: call synthesize() from one worker."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.model = None

    def load(self) -> None:
        from mlx_audio.tts.utils import load

        # Always load from a local snapshot of the pinned revision: cached, or
        # downloaded first (only reached with TTS_AUTO_DOWNLOAD — the daemon
        # gates on model_cached() otherwise).
        model = load(cached_model_path() or download_model())
        if model.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"expected native {SAMPLE_RATE} Hz output, model reports {model.sample_rate} Hz"
            )
        self.model = model

    def synthesize(
        self,
        text: str,
        language: str = "auto",
        instruction: str = "",
        cancel=None,
        progress=None,
    ) -> tuple[np.ndarray, dict]:
        """Synthesize text to float32 mono audio at SAMPLE_RATE.

        cancel: optional () -> bool, checked before each chunk; True raises JobCancelled.
        progress: optional (done, total, meta) -> None, called after resolving chunks
                  (done=0) and after each chunk.
        """
        if self.model is None:
            raise RuntimeError("model not loaded")

        if language == "auto":
            language = detect_language(text)
        speaker = VOICES[language]

        paragraphs = chunk_text(text)
        total = sum(len(p) for p in paragraphs)
        meta = {"voice": speaker, "language": language, "chunks_total": total}
        if progress:
            progress(0, total, meta)

        para_pause = np.zeros(int(SAMPLE_RATE * PARAGRAPH_PAUSE_MS / 1000), dtype=np.float32)
        chunk_pause = np.zeros(int(SAMPLE_RATE * CHUNK_PAUSE_MS / 1000), dtype=np.float32)

        segments: list[np.ndarray] = []
        done = 0
        for p_idx, chunks in enumerate(paragraphs):
            if p_idx > 0:
                segments.append(para_pause)
            for c_idx, chunk in enumerate(chunks):
                if cancel and cancel():
                    raise JobCancelled()
                if c_idx > 0:
                    segments.append(chunk_pause)
                cap_sec = max(CAP_SEC_FLOOR, len(chunk) * CAP_SEC_PER_CHAR)
                audio_parts = []
                for r in self.model.generate_custom_voice(
                    text=chunk,
                    speaker=speaker,
                    language=language,
                    instruct=instruction or None,
                    max_tokens=int(cap_sec * CODEC_TOKENS_PER_SEC),
                ):
                    if cancel and cancel():
                        raise JobCancelled()
                    audio_parts.append(np.asarray(r.audio, dtype=np.float32))
                chunk_audio = np.concatenate(audio_parts)
                if len(chunk_audio) >= (cap_sec - 1.0) * SAMPLE_RATE:
                    log.warning(
                        "runaway generation capped at %.1fs for %d-char chunk %r",
                        len(chunk_audio) / SAMPLE_RATE, len(chunk), chunk[:40],
                    )
                segments.append(chunk_audio)
                done += 1
                if progress:
                    progress(done, total, meta)

        if not segments:
            raise ValueError("no synthesizable text")
        return np.concatenate(segments), meta


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def encode_m4a(wav_path: Path, m4a_path: Path) -> None:
    """Encode WAV to AAC-in-M4A with macOS's built-in afconvert."""
    result = subprocess.run(
        [
            "afconvert",
            "-f", "m4af",
            "-d", "aac",
            "-b", str(M4A_BITRATE),
            str(wav_path),
            str(m4a_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"afconvert failed: {result.stderr.strip() or result.stdout.strip()}")
