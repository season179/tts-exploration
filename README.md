# Local TTS (Qwen3 CustomVoice on MLX)

Local-only text-to-speech for Apple Silicon, callable by other applications
and AI agents. Uses `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` via
`mlx-audio`. Voices: **Aiden** (English), **Serena** (Chinese).

One installable package (`tts_local/`), three parts:

- **`cli.py`** — the `tts` command (thin client, fast startup)
- **`daemon.py`** — resident inference daemon (HTTP API on loopback)
- **`core.py`** — synthesis engine (model, chunking, encoding)

The model (~3.5 GB) loads once into the daemon and stays resident; every
call after that is pure inference time.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/season179/tts-exploration/main/install.sh | sh
tts setup   # one-time: compatibility check + ~3.5 GB model download
```

The installer needs git (Xcode Command Line Tools) and installs
[uv](https://docs.astral.sh/uv/) if missing, then `uv tool install`s this
repo. `tts setup` is idempotent — safe to re-run any time. The daemon never
downloads the model on its own: without setup you get a clear error and the
suggestion to run it (agents can pass `--auto-download` to opt in instead).

From a checkout (development):

```bash
uv tool install --force .          # install your working copy as `tts`
uv run tts daemon start --foreground   # or run the daemon in this terminal
```

## CLI usage

```bash
tts speak "Hello there" -o hello.m4a          # auto-starts the daemon
echo "Hello" | tts speak -o hello.m4a         # stdin works too
tts speak "你好" --format wav -o hello.wav    # WAV instead of M4A
tts speak "Hi" --json                         # machine-readable result
```

Default output format is **M4A** (AAC 64 kbps, encoded with macOS's built-in
`afconvert`, ~7× smaller than WAV). Pass `--format wav` for uncompressed
24 kHz mono PCM.

Async workflow:

```bash
id=$(tts submit "Long text ...")
tts status "$id" --json
tts wait "$id" -o out.m4a
tts cancel "$id"
```

Daemon management (usually unnecessary — `speak`/`submit` auto-start it):

```bash
tts daemon start|stop|status|logs
tts health --json
tts voices --json
```

Useful flags: `--language auto|english|chinese`, `--instruction "Calm, warm
narration."` (max 18 words; 36 chars Chinese), `-o -` (binary audio to
stdout), `--no-start`, `--timeout`, `--force`, `--quiet`. Keep instructions to
emotion + pace + 1–2 vocal qualities, phrased as an imperative or compact
descriptor.

Exit codes: `0` ok · `2` usage · `3` daemon unavailable · `4` timeout ·
`5` busy · `6` synthesis failed/canceled · `7` output I/O · `130` interrupted.

`--json` prints exactly one JSON object on stdout; progress and logs go to
stderr.

## Web UI

Open the **web UI URL printed by `tts daemon status`** — it ends in
`#<token>`, which the page uses to authenticate API calls. The fragment
never leaves the browser, and the page itself contains no secrets. Paste
text, pick language/format, optional style instruction, per-chunk progress,
inline playback and download.

## HTTP API (for applications)

State lives in `~/.local/state/qwen-tts/` (override with `TTS_STATE_DIR`).
`daemon.json` there (mode 0600) holds `{pid, port, token,
protocol_version}` — read it to discover the endpoint and bearer token.
All endpoints except `/v1/health` require `Authorization: Bearer <token>`.

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/health` | `loading` / `ready` / `failed`, queue depth (no auth) |
| `POST /v1/jobs` | `{text, language?, instruction?, format?}` → `202 {id}` |
| `GET /v1/jobs/{id}` | status, progress, and `audio_url` when done |
| `DELETE /v1/jobs/{id}` | cancel (queued instantly; running between chunks) |
| `GET /v1/jobs/{id}/audio` | the audio file (`audio/mp4` or `audio/wav`) |
| `GET /v1/voices` | voices, languages, formats, limits |
| `POST /v1/shutdown` | graceful stop |

Errors are `{"error": {"code", "message", "retryable"}}`; `429` (queue full,
capacity 16) and `503` (model loading) include `Retry-After`. Finished jobs
and their audio expire after 1 hour. Jobs do not survive a daemon restart.

## Behavior notes

- Text is split into paragraphs (blank lines) and sentence chunks of ~300–400
  chars; language `auto` picks Serena when the text is ≥30% CJK, else Aiden.
- No style instruction is applied by default. Instructions are capped at 18
  words (36 chars Chinese) because long prose destabilizes generation. On
  short or Chinese text they can still trigger runaway generation (the model
  fails to stop speaking). A per-chunk guard caps the damage at
  0.6 s per character (10 s minimum) and logs a warning when it fires, so a
  runaway costs seconds, not minutes.
- One job synthesizes at a time (single MLX worker); others queue.
- The daemon binds its port immediately and reports `loading` until the model
  is ready, so callers can distinguish "starting" from "absent".
- A non-blocking lock guarantees a single daemon; a second start exits
  immediately. If port 8765 is taken the daemon falls back to an ephemeral
  port — always discover via `daemon.json`, don't hard-code the port.
