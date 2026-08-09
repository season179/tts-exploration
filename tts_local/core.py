"""Synthesis engine for the local Qwen3-TTS daemon.

Pure inference layer: model loading, text chunking, synthesis, and audio
encoding. No HTTP or job-queue concerns.
"""

import logging
import subprocess
import wave
from pathlib import Path

import mlx.core as mx
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
MAX_INSTRUCTION_WORDS = 18
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
# Watchdog silence threshold. Measured on real output: quietest genuine-speech
# 2s chunk (soft instructed delivery) is -31 dB RMS; runaway junk is -70 dB but
# can include low-level noise, so -60 dB missed some trajectories. -40 dB keeps
# a 3x amplitude margin below real speech.
SILENCE_RMS = 1e-2

VOICES = {
    "Ryan": {"language": "english"},
    "Aiden": {"language": "english"},
    "Vivian": {"language": "chinese"},
    "Serena": {"language": "chinese"},
    "Uncle_Fu": {"language": "chinese"},
    "Dylan": {"language": "chinese", "note": "Beijing dialect"},
    "Eric": {"language": "chinese", "note": "Sichuan dialect"},
}
DEFAULT_VOICES = {"english": "Ryan", "chinese": "Serena"}
LANGUAGES = tuple(DEFAULT_VOICES)
FORMATS = ("m4a", "wav")

# CJK punctuation, unified ideographs (+ext A), compatibility ideographs,
# and half/fullwidth forms. Kept as \u escapes: some endpoints have homoglyphs.
CJK_RE = re.compile("[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

# Sentence boundary: split after terminal punctuation (+ optional closing quotes).
SENTENCE_RE = re.compile(r"[^.!?。！？；;…\n]*[.!?。！？；;…]+[\"'”’）)\]]*|[^.!?。！？；;…\n]+")


class JobCancelled(Exception):
    """Raised inside synthesize() when the cancel callback returns True."""


def valid_voices_text() -> str:
    """Human-readable speaker catalogue for validation errors."""
    return ", ".join(
        f"{name} ({metadata['language']})" for name, metadata in VOICES.items()
    )


def validate_voice(voice: str | None) -> None:
    """Raise a teaching error unless voice is omitted or names a model speaker."""
    if voice is not None and (not isinstance(voice, str) or voice not in VOICES):
        raise ValueError(
            f"unknown voice {voice!r}. Valid voices: {valid_voices_text()}. "
            "Voices may be used with either language; run `tts voices` for details."
        )


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
        voice: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Synthesize text to float32 mono audio at SAMPLE_RATE.

        cancel: optional () -> bool, checked before each chunk; True raises JobCancelled.
        progress: optional (done, total, meta) -> None, called after resolving chunks
                  (done=0) and after each chunk.
        """
        if self.model is None:
            raise RuntimeError("model not loaded")

        validate_voice(voice)
        if language == "auto":
            language = detect_language(text)
        speaker = voice or DEFAULT_VOICES[language]

        paragraphs = chunk_text(text)
        total = sum(len(p) for p in paragraphs)
        meta = {"voice": speaker, "language": language, "chunks_total": total}
        if progress:
            progress(0, total, meta)

        para_pause = np.zeros(int(SAMPLE_RATE * PARAGRAPH_PAUSE_MS / 1000), dtype=np.float32)
        chunk_pause = np.zeros(int(SAMPLE_RATE * CHUNK_PAUSE_MS / 1000), dtype=np.float32)

        retry_counter = 0

        def generate_attempt(
            attempt_text: str, attempt_instruction: str | None, cap_sec: float,
            reseed: bool = False,
        ) -> tuple[np.ndarray, str | None]:
            nonlocal retry_counter
            if reseed:
                retry_counter += 1
                mx.random.seed(retry_counter)
                log.warning("reseeded retry with seed %d", retry_counter)

            audio_parts: list[np.ndarray] = []
            speech_seen = False
            silent_chunks = 0
            for r in self.model.generate_custom_voice(
                text=attempt_text,
                speaker=speaker,
                language=language,
                instruct=attempt_instruction,
                max_tokens=int(cap_sec * CODEC_TOKENS_PER_SEC),
                stream=True,
                streaming_interval=2.0,
            ):
                if cancel and cancel():
                    raise JobCancelled()
                part = np.asarray(r.audio, dtype=np.float32)
                audio_parts.append(part)
                rms = float(np.sqrt(np.mean(part * part))) if part.size else 0.0
                if rms >= SILENCE_RMS:
                    speech_seen = True
                    silent_chunks = 0
                elif speech_seen:
                    silent_chunks += 1
                    if silent_chunks == 3:
                        trailing_sec = sum(
                            len(p) for p in audio_parts[-silent_chunks:]
                        ) / SAMPLE_RATE
                        total_sec = sum(len(p) for p in audio_parts) / SAMPLE_RATE
                        del audio_parts[-silent_chunks:]
                        log.warning(
                            "silence watchdog detected runaway at %.1fs after %.1fs "
                            "of trailing silence for %d-char text %r",
                            total_sec, trailing_sec, len(attempt_text), attempt_text[:40],
                        )
                        return np.concatenate(audio_parts), "silence watchdog"

            if not audio_parts:
                log.warning("generation produced no audio for text %r", attempt_text[:40])
                return np.zeros(0, dtype=np.float32), "empty output"
            audio = np.concatenate(audio_parts)
            if len(audio) >= (cap_sec - 1.0) * SAMPLE_RATE:
                log.warning(
                    "runaway generation capped at %.1fs for %d-char text %r",
                    len(audio) / SAMPLE_RATE, len(attempt_text), attempt_text[:40],
                )
                return audio, "token cap"
            return audio, None

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
                chunk_audio, runaway = generate_attempt(
                    chunk, instruction or None, cap_sec
                )
                if runaway:
                    log.warning(
                        "%s triggered; retrying same chunk with instruction and new seed",
                        runaway,
                    )
                    chunk_audio, runaway = generate_attempt(
                        chunk, instruction or None, cap_sec, reseed=True
                    )
                if runaway:
                    log.warning(
                        "reseeded instruction retry also hit %s; "
                        "retrying without instruction",
                        runaway,
                    )
                    chunk_audio, runaway = generate_attempt(
                        chunk, None, cap_sec, reseed=True
                    )
                if runaway:
                    sentences = split_sentences(chunk)
                    log.warning(
                        "retry without instruction also hit %s; "
                        "splitting chunk into %d sentences",
                        runaway, len(sentences),
                    )
                    sentence_segments: list[np.ndarray] = []
                    for s_idx, sentence in enumerate(sentences):
                        if cancel and cancel():
                            raise JobCancelled()
                        if s_idx > 0:
                            sentence_segments.append(chunk_pause)
                        sentence_cap_sec = max(
                            CAP_SEC_FLOOR, len(sentence) * CAP_SEC_PER_CHAR
                        )
                        sentence_audio, sentence_runaway = generate_attempt(
                            sentence, None, sentence_cap_sec, reseed=True
                        )
                        if sentence_runaway:
                            log.warning(
                                "sentence fallback hit %s for %d-char sentence %r; "
                                "keeping audio",
                                sentence_runaway, len(sentence), sentence[:40],
                            )
                        sentence_segments.append(sentence_audio)
                    chunk_audio = np.concatenate(sentence_segments)
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
