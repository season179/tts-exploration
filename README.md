# Local TTS (Qwen3 CustomVoice on MLX)

Local-only text-to-speech web app for Apple Silicon. Uses
`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` via `mlx-audio`.
Voices: **Aiden** (English), **Serena** (Chinese).

## Setup

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python app.py
```

Open <http://127.0.0.1:8765>. First run downloads the model (~3.5 GB) to the
Hugging Face cache; model load takes ~1 minute, then stays resident.

## Usage

Paste text (up to 10,000 chars), pick Auto / English / Chinese, optionally add
a style instruction (e.g. "Speak slowly and calmly."), hit Generate. Progress
shows per-chunk; result plays inline and downloads as 24 kHz mono WAV.

- **Auto** picks Serena when the text is ≥30% CJK, otherwise Aiden.
- Text is split into paragraphs (blank lines) and sentence chunks of ~300–400
  chars; 250 ms pauses between paragraphs, 80 ms between chunks.
- One request at a time — inference is serialized through a single worker
  queue; extra requests wait with a queue position shown.

## API

- `POST /speak` — JSON `{text, language: auto|english|chinese, instruction?}` → `{id}`
- `GET /status/<id>` — job state, chunk progress, timing
- `GET /audio/<id>.wav` — finished audio
- Server binds `127.0.0.1:8765` only.

Generated audio lands in `outputs/` (gitignored).
