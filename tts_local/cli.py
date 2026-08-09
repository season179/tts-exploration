#!/usr/bin/env python3
"""tts — CLI client for the local Qwen3-TTS daemon.

Talks to the daemon (tts_local.daemon) over loopback HTTP using the
discovery file in the state dir; auto-starts the daemon when needed.
Heavy imports stay lazy: every command except `setup` is stdlib-only.

Exit codes:
  0 success            4 timed out           7 output I/O error
  2 usage error        5 daemon busy (429)   130 interrupted
  3 daemon unavailable 6 synthesis failed/canceled
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL_VERSION = 1
STATE_DIR = Path(os.environ.get("TTS_STATE_DIR", "~/.local/state/qwen-tts")).expanduser()
DISCOVERY_PATH = STATE_DIR / "daemon.json"
LOG_PATH = STATE_DIR / "daemon.log"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_TIMEOUT = 4
EXIT_BUSY = 5
EXIT_FAILED = 6
EXIT_OUTPUT = 7
EXIT_INTERRUPT = 130

POLL_INTERVAL = 0.5


class CliError(Exception):
    def __init__(self, code: int, message: str, error_code: str = "error"):
        super().__init__(message)
        self.code = code
        self.error_code = error_code


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- daemon API


def read_discovery() -> dict | None:
    try:
        data = json.loads(DISCOVERY_PATH.read_text())
    except (OSError, ValueError):
        return None
    if not all(k in data for k in ("pid", "port", "token", "protocol_version")):
        return None
    return data


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def request(d: dict, method: str, path: str, body: dict | None = None,
            timeout: float = 30, auth: bool = True, raw: bool = False):
    url = f"http://127.0.0.1:{d['port']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if auth:
        req.add_header("Authorization", f"Bearer {d['token']}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            return payload if raw else json.loads(payload)
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read())["error"]
        except (ValueError, KeyError):
            err = {"code": "http_error", "message": f"HTTP {exc.code}"}
        if exc.code == 429:
            raise CliError(EXIT_BUSY, err["message"], err["code"])
        if exc.code == 503:
            raise CliError(EXIT_UNAVAILABLE, err["message"], err["code"])
        raise CliError(EXIT_FAILED, err["message"], err["code"])


def probe(d: dict, timeout: float = 3) -> dict | None:
    """Health-check a discovery record; None means daemon not usable."""
    try:
        health = request(d, "GET", "/v1/health", timeout=timeout, auth=False)
    except (CliError, OSError):
        return None
    # After a crash another process could squat the old port; only trust a
    # responder whose pid matches the (0600) discovery record.
    if health.get("pid") != d["pid"]:
        return None
    if health.get("protocol_version") != PROTOCOL_VERSION:
        raise CliError(
            EXIT_UNAVAILABLE,
            f"daemon speaks protocol v{health.get('protocol_version')}, "
            f"this CLI expects v{PROTOCOL_VERSION}; run: tts daemon stop && tts daemon start",
            "protocol_mismatch",
        )
    return health


def daemon_python() -> str:
    # The daemon lives in the same installed environment as this CLI.
    return os.environ.get("TTS_DAEMON_PYTHON", sys.executable)


def daemon_argv() -> list[str]:
    return [daemon_python(), "-m", "tts_local.daemon"]


def spawn_daemon(auto_download: bool = False) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    env = dict(os.environ)
    if auto_download:
        env["TTS_AUTO_DOWNLOAD"] = "1"
    with open(LOG_PATH, "ab") as logf:
        subprocess.Popen(
            daemon_argv(),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
            env=env,
        )


def connect(no_start: bool = False, start_timeout: float = 180,
            need_ready: bool = True, auto_download: bool = False) -> tuple[dict, dict]:
    """Return (discovery, health) for a usable daemon, starting one if allowed."""
    d = read_discovery()
    health = probe(d) if d else None
    if health is None:
        # A stale discovery file is left in place: unlinking here races a
        # freshly started daemon that just rewrote it. The daemon overwrites
        # it on startup and probe() rejects stale records anyway.
        if no_start:
            raise CliError(EXIT_UNAVAILABLE, "daemon not running (--no-start given)",
                           "daemon_unavailable")
        log_err("starting tts daemon (first call loads the model, ~1 min) ...")
        spawn_daemon(auto_download=auto_download)

    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        d = read_discovery()
        health = probe(d) if d else None
        if health:
            if health["status"] == "failed":
                raise CliError(EXIT_UNAVAILABLE,
                               f"daemon model load failed: {health.get('error')}",
                               "model_failed")
            if health["status"] == "ready" or not need_ready:
                return d, health
        time.sleep(1.0)
    raise CliError(EXIT_TIMEOUT, f"daemon not ready within {start_timeout:.0f}s "
                   f"(see {LOG_PATH})", "start_timeout")


# ---------------------------------------------------------------- output


def emit(args, payload: dict, human: str | None = None) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload))
    elif human:
        print(human)


def fail(args, exc: CliError) -> int:
    if getattr(args, "json", False):
        print(json.dumps({"ok": False,
                          "error": {"code": exc.error_code, "message": str(exc)}}))
    log_err(f"tts: {exc}")
    return exc.code


def read_text_input(args) -> str:
    if getattr(args, "text", None):
        return args.text
    if sys.stdin.isatty() and not args.stdin:
        raise CliError(EXIT_USAGE, "no text: pass TEXT, pipe stdin, or use --stdin", "usage")
    return sys.stdin.read()


def download(d: dict, job: dict, dest: str, force: bool) -> str | None:
    """Fetch job audio to dest ('-' = stdout). Returns final path or None."""
    data = request(d, "GET", f"/v1/jobs/{job['id']}/audio", raw=True, timeout=120)
    if dest == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return None
    target = Path(dest)
    if target.exists() and not force:
        raise CliError(EXIT_OUTPUT, f"{target} exists (use --force to overwrite)",
                       "output_exists")
    tmp = target.with_name(f".{target.name}.part{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise CliError(EXIT_OUTPUT, f"cannot write {target}: {exc}", "output_io")
    return str(target.resolve())


def poll_until_done(d: dict, job_id: str, timeout: float, show_progress: bool) -> dict:
    deadline = time.monotonic() + timeout
    last_line = ""
    while True:  # always poll at least once, even with --timeout 0
        job = request(d, "GET", f"/v1/jobs/{job_id}")
        status = job["status"]
        if show_progress and status in ("queued", "running"):
            if status == "queued":
                line = f"queued (position {job.get('queue_position', 0) + 1})"
            else:
                line = (f"generating with {job.get('voice', '?')}: "
                        f"chunk {job.get('chunks_done', 0)}/{job.get('chunks_total', '?')}")
            if line != last_line:
                log_err(line)
                last_line = line
        if status in ("done", "failed", "canceled"):
            return job
        if time.monotonic() >= deadline:
            raise CliError(EXIT_TIMEOUT, f"job {job_id} still {status} after {timeout:.0f}s "
                           "(it keeps running; use tts wait/status/cancel)", "timeout")
        time.sleep(POLL_INTERVAL)


def finish_job(args, d: dict, job: dict, dest: str | None) -> int:
    if job["status"] in ("failed", "canceled"):
        err = job.get("error") or {}
        raise CliError(EXIT_FAILED,
                       err.get("message", f"job {job['status']}"),
                       err.get("code", job["status"]))
    path = download(d, job, dest, args.force) if dest else None
    payload = {
        "ok": True,
        "id": job["id"],
        "path": path,
        "format": job["format"],
        "duration_sec": job.get("duration_sec"),
        "generation_sec": job.get("generation_sec"),
        "sample_rate": job.get("sample_rate"),
        "voice": job.get("voice"),
        "language": job.get("resolved_language"),
    }
    emit(args, payload, human=path or "")
    return EXIT_OK


# ---------------------------------------------------------------- commands


def submit_job(args, d: dict) -> dict:
    text = read_text_input(args)
    body = {"text": text, "language": args.language, "format": args.format}
    if args.instruction is not None:
        body["instruction"] = args.instruction
    return request(d, "POST", "/v1/jobs", body=body)


def effective_start_timeout(args) -> float:
    if args.start_timeout is not None:
        return args.start_timeout
    # A first-time model download rides on this deadline; give it real time.
    return 1800 if getattr(args, "auto_download", False) else 180


def cmd_speak(args) -> int:
    if args.output == "-" and args.json:
        raise CliError(EXIT_USAGE, "-o - and --json both claim stdout; pick one", "usage")
    d, _ = connect(args.no_start, effective_start_timeout(args),
                   auto_download=args.auto_download)
    accepted = submit_job(args, d)
    job_id = accepted["id"]
    show_progress = sys.stderr.isatty() and not args.quiet
    try:
        job = poll_until_done(d, job_id, args.timeout, show_progress)
    except KeyboardInterrupt:
        if args.json:
            print(json.dumps({"ok": False, "id": job_id,
                              "error": {"code": "interrupted",
                                        "message": "interrupted; job keeps running"}}))
        log_err(f"interrupted; job {job_id} keeps running (tts cancel {job_id} to stop it)")
        return EXIT_INTERRUPT
    dest = args.output or f"tts-{job_id[:8]}.{args.format}"
    return finish_job(args, d, job, dest)


def cmd_submit(args) -> int:
    d, _ = connect(args.no_start, effective_start_timeout(args),
                   auto_download=args.auto_download)
    accepted = submit_job(args, d)
    emit(args, {"ok": True, **accepted}, human=accepted["id"])
    return EXIT_OK


def cmd_status(args) -> int:
    d, _ = connect(no_start=True, need_ready=False)
    job = request(d, "GET", f"/v1/jobs/{args.job_id}")
    emit(args, {"ok": True, **job},
         human=f"{job['status']}"
               + (f" chunk {job.get('chunks_done', 0)}/{job.get('chunks_total', '?')}"
                  if job["status"] == "running" else ""))
    return EXIT_OK


def cmd_wait(args) -> int:
    if args.output == "-" and args.json:
        raise CliError(EXIT_USAGE, "-o - and --json both claim stdout; pick one", "usage")
    d, _ = connect(no_start=True, need_ready=False)
    job = poll_until_done(d, args.job_id, args.timeout,
                          sys.stderr.isatty() and not args.quiet)
    return finish_job(args, d, job, args.output)


def cmd_cancel(args) -> int:
    d, _ = connect(no_start=True, need_ready=False)
    job = request(d, "DELETE", f"/v1/jobs/{args.job_id}")
    emit(args, {"ok": True, **job}, human=job["status"])
    return EXIT_OK


def cmd_health(args) -> int:
    d = read_discovery()
    health = probe(d) if d else None
    if health is None:
        raise CliError(EXIT_UNAVAILABLE, "daemon not running", "daemon_unavailable")
    emit(args, {"ok": True, **health},
         human=f"{health['status']} (pid {health['pid']}, "
               f"port {d['port']}, queued {health['jobs_queued']})")
    return EXIT_OK


def cmd_voices(args) -> int:
    d, _ = connect(no_start=True, need_ready=False)
    info = request(d, "GET", "/v1/voices")
    emit(args, {"ok": True, **info},
         human="\n".join(f"{v['name']}\t{v['language']}" for v in info["voices"]))
    return EXIT_OK


def cmd_daemon_start(args) -> int:
    if args.foreground:
        if args.json:
            raise CliError(EXIT_USAGE, "--foreground and --json cannot be combined", "usage")
        if args.auto_download:
            os.environ["TTS_AUTO_DOWNLOAD"] = "1"
        argv = daemon_argv()
        os.execvp(argv[0], argv)
    d = read_discovery()
    health = probe(d) if d else None
    already = bool(health and health["status"] == "ready")
    if not already:
        # Not running, still loading, or failed: connect() spawns only if
        # unreachable, then waits for ready (raising on a failed model load).
        d, health = connect(no_start=False, start_timeout=effective_start_timeout(args),
                            auto_download=args.auto_download)
    url = f"http://127.0.0.1:{d['port']}/#{d['token']}"
    emit(args, {"ok": True, "already_running": already,
                "pid": health["pid"], "port": d["port"], "ui_url": url},
         human=f"ready (pid {health['pid']}, {url})")
    return EXIT_OK


def cmd_daemon_stop(args) -> int:
    d = read_discovery()
    if not d:
        emit(args, {"ok": True, "was_running": False}, human="not running")
        return EXIT_OK
    if probe(d):
        request(d, "POST", "/v1/shutdown")
    elif pid_alive(d["pid"]):
        raise CliError(EXIT_UNAVAILABLE,
                       f"pid {d['pid']} alive but not answering; kill it manually",
                       "daemon_unresponsive")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and pid_alive(d["pid"]):
        time.sleep(0.3)
    if pid_alive(d["pid"]):
        raise CliError(EXIT_UNAVAILABLE, f"daemon (pid {d['pid']}) did not exit", "stop_failed")
    # No discovery cleanup here: the daemon removes its own file on graceful
    # shutdown, probe() rejects stale records, and unlinking from the client
    # races a fresh daemon publishing its record.
    emit(args, {"ok": True, "was_running": True}, human="stopped")
    return EXIT_OK


def cmd_daemon_status(args) -> int:
    d = read_discovery()
    health = probe(d) if d else None
    if health is None:
        emit(args, {"ok": True, "running": False}, human="not running")
        return EXIT_UNAVAILABLE
    url = f"http://127.0.0.1:{d['port']}/#{d['token']}"  # fragment = UI access token
    emit(args, {"ok": True, "running": True, "url": f"http://127.0.0.1:{d['port']}",
                "ui_url": url, **health},
         human=(f"{health['status']} — pid {health['pid']}, "
                f"model {health['model_id']}, up {health['uptime_sec']:.0f}s, "
                f"queued {health['jobs_queued']}/{health['queue_capacity']}\n"
                f"web UI: {url}"))
    return EXIT_OK


def cmd_setup(args) -> int:
    """Check machine compatibility, then download the model if missing."""
    import platform
    import shutil

    problems: list[str] = []
    warnings: list[str] = []

    if sys.platform != "darwin":
        problems.append(f"macOS required (this is {sys.platform}); "
                        "the MLX runtime only exists for Apple platforms")
    if platform.machine() != "arm64":
        problems.append(f"Apple Silicon (arm64) required, found {platform.machine()}; "
                        "MLX does not support Intel Macs")
    if not shutil.which("afconvert"):
        problems.append("afconvert not found (ships with macOS; needed for M4A output)")

    ram_gb = None
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        ram_gb = int(out.stdout.strip()) / 2**30
        if ram_gb < 8:
            problems.append(f"{ram_gb:.0f} GB RAM is not enough (the model needs "
                            "~4 GB resident; 8 GB minimum, 16 GB recommended)")
        elif ram_gb < 16:
            warnings.append(f"{ram_gb:.0f} GB RAM: works, but expect memory pressure "
                            "alongside other apps (16 GB recommended)")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        warnings.append("could not determine RAM size")

    if problems:
        payload = {"ok": False, "problems": problems, "warnings": warnings}
        if args.json:
            print(json.dumps(payload))
        for p in problems:
            log_err(f"tts: incompatible: {p}")
        return EXIT_UNAVAILABLE

    # Heavy imports only past the cheap checks.
    from huggingface_hub.constants import HF_HUB_CACHE

    from tts_local.core import MODEL_ID, download_model, model_cached

    cache_root = Path(HF_HUB_CACHE)
    probe_dir = cache_root
    while not probe_dir.exists():
        probe_dir = probe_dir.parent
    free_gb = shutil.disk_usage(probe_dir).free / 2**30

    present = model_cached()
    downloaded_now = False
    if not present:
        if free_gb < 5:
            msg = (f"only {free_gb:.1f} GB free at {cache_root}; "
                   "the model needs ~3.5 GB (5 GB to be safe)")
            if args.json:
                print(json.dumps({"ok": False, "problems": [msg], "warnings": warnings}))
            log_err(f"tts: {msg}")
            return EXIT_UNAVAILABLE
        log_err(f"downloading {MODEL_ID} (~3.5 GB, one-time) ...")
        try:
            download_model()
        except Exception as exc:
            msg = f"model download failed: {exc}"
            if args.json:
                print(json.dumps({"ok": False, "problems": [msg], "warnings": warnings}))
            log_err(f"tts: {msg}")
            return EXIT_UNAVAILABLE
        downloaded_now = True

    # A daemon that started before the model existed is wedged in "failed";
    # stop it so the next call starts fresh against the now-present model.
    stopped_failed = False
    d = read_discovery()
    if d:
        try:
            health = request(d, "GET", "/v1/health", timeout=3, auth=False)
            if health.get("status") == "failed" and health.get("pid") == d.get("pid"):
                request(d, "POST", "/v1/shutdown")
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and pid_alive(d["pid"]):
                    time.sleep(0.3)
                stopped_failed = not pid_alive(d["pid"])
                log_err("stopped a previously-failed daemon; the next call starts fresh"
                        if stopped_failed else
                        f"warning: failed daemon (pid {d['pid']}) has not exited yet")
        except Exception:
            pass

    for w in warnings:
        log_err(f"tts: note: {w}")
    emit(args, {"ok": True, "model_id": MODEL_ID, "model_downloaded": True,
                "downloaded_now": downloaded_now, "stopped_failed_daemon": stopped_failed,
                "ram_gb": round(ram_gb) if ram_gb else None, "warnings": warnings},
         human=("model downloaded — ready to use: try  tts speak \"hello\" -o hello.m4a"
                if downloaded_now else "already set up — nothing to do"))
    return EXIT_OK


AGENTS_DOC = """\
# tts — local text-to-speech CLI (for scripts and AI agents)

Offline Qwen3-TTS on Apple Silicon. Two voices: Aiden (English), Serena
(Chinese); language is auto-detected. Output: M4A (default) or WAV
(24 kHz mono). A resident daemon holds the model; the CLI auto-starts it
(first start ~1 min to load the model, later calls take seconds).

## Synchronous (most common)

    tts speak "text to say" -o out.m4a --json
    echo "text" | tts speak -o out.m4a --json      # stdin
    tts speak "text" -o - > out.m4a                # binary audio to stdout

Options: --instruction "Calm, warm narration." — max 18 words (36 chars Chinese),
--language auto|english|chinese, --format m4a|wav, --timeout SECONDS,
--force (overwrite), --quiet, --no-start (fail instead of starting daemon).
Max 10,000 chars per request; long text is chunked automatically.

## Async (long texts, parallel work)

    id=$(tts submit "long text ...")
    tts status "$id" --json          # queued | running | done | failed | canceled
    tts wait "$id" -o out.m4a --json
    tts cancel "$id" --json

## Introspection / lifecycle

    tts health --json                # daemon state without starting it
    tts voices --json                # voices, languages, formats, limits
    tts daemon status|start|stop|logs
    tts setup                        # one-time: hardware check + ~3.5 GB model download

## Contract

- --json: exactly one JSON object on stdout; progress and logs on stderr.
  Success objects have "ok": true; errors {"ok": false, "error": {code, message}}.
- Exit codes: 0 ok · 2 usage · 3 daemon unavailable (incl. model not
  downloaded — run `tts setup`) · 4 timeout · 5 busy queue · 6 synthesis
  failed/canceled · 7 output I/O · 130 interrupted.
- Direct HTTP API: read ~/.local/state/qwen-tts/daemon.json (0600) for
  {port, token}; bearer-auth /v1 endpoints (health, jobs, voices). Never
  hard-code port 8765.

## Caveats

- Keep --instruction short: emotion + pace + 1-2 vocal qualities, as an
  imperative or compact descriptor. Examples: "Calm, warm narration." or
  "Speak in a frightened, trembling tone, voice shaky."
- A style --instruction on very short text can rarely produce overlong
  audio; a built-in guard caps it (~0.6 s/char, min 10 s).
- One instruction applies to the whole request; split text needing
  different styles into separate calls.
"""


def cmd_agents(args) -> int:
    print(AGENTS_DOC, end="")
    return EXIT_OK


def cmd_daemon_logs(args) -> int:
    if not LOG_PATH.exists():
        log_err(f"no log file at {LOG_PATH}")
        return EXIT_UNAVAILABLE
    lines = LOG_PATH.read_text(errors="replace").splitlines()
    print("\n".join(lines[-args.lines:]))
    return EXIT_OK


# ---------------------------------------------------------------- argparse


def add_input_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("text", nargs="?", help="text to speak (default: read stdin)")
    p.add_argument("--stdin", action="store_true", help="force reading text from stdin")
    p.add_argument("--language", default="auto", choices=["auto", "english", "chinese"])
    p.add_argument(
        "--instruction", help="short style cue; max 18 words (36 chars Chinese)"
    )
    p.add_argument("--format", default="m4a", choices=["m4a", "wav"],
                   help="audio format (default m4a)")


def add_daemon_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-start", action="store_true", help="fail if daemon is not running")
    p.add_argument("--start-timeout", type=float, default=None,
                   help="seconds to wait for daemon readiness "
                        "(default 180, or 1800 with --auto-download)")
    p.add_argument("--auto-download", action="store_true",
                   help="allow the daemon to download the model (~3.5 GB) if missing")


EXAMPLES = """\
examples:
  tts speak "hello there" -o hello.m4a
  echo "piped text" | tts speak -o out.m4a
  tts speak "Once upon a time, a little fox lost his way." \\
      --instruction "Warm, gentle bedtime-story narration." -o story.m4a
  tts speak "你好，很高兴认识你。" --format wav -o zh.wav
  tts speak "hi" --json                      # machine-readable result
  id=$(tts submit "long text ..."); tts wait "$id" -o long.m4a
  tts daemon status                          # daemon state + web UI URL

speak/submit options (see `tts speak --help` for all):
  --instruction TEXT   short style cue; max 18 words (36 chars Chinese)
  --language X         auto | english | chinese        --format X   m4a | wav

for scripts and agents:
  - run `tts agents` for the full machine-oriented usage guide
  - --json prints exactly one JSON object on stdout; progress/logs go to stderr
  - the daemon auto-starts on first use (~1 min model load; later calls take
    seconds and the daemon stays resident)
  - one-time setup on a new machine: `tts setup` (checks hardware, downloads
    the ~3.5 GB model); until then commands exit 3 with a message saying so
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tts", description=__doc__, epilog=EXAMPLES,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("speak", help="synthesize text and wait for the audio")
    add_input_opts(p)
    add_daemon_opts(p)
    p.add_argument("-o", "--output", help="output file, or - for binary stdout "
                                          "(default tts-<id>.<format> in cwd)")
    p.add_argument("--force", action="store_true", help="overwrite existing output file")
    p.add_argument("--timeout", type=float, default=600,
                   help="seconds to wait for synthesis (default 600)")
    p.add_argument("--quiet", action="store_true", help="no progress on stderr")
    p.add_argument("--json", action="store_true", help="machine-readable result on stdout")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("submit", help="enqueue a job, print its id immediately")
    add_input_opts(p)
    add_daemon_opts(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="show job status")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("wait", help="wait for a job, optionally save its audio")
    p.add_argument("job_id")
    p.add_argument("-o", "--output", help="output file, or - for binary stdout")
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("cancel", help="cancel a queued or running job")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("agents", help="print the usage guide for scripts and AI agents")
    p.set_defaults(func=cmd_agents)

    p = sub.add_parser("setup", help="check machine compatibility and download "
                                     "the model (idempotent)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("health", help="daemon health")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("voices", help="list voices, languages, formats")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_voices)

    pd = sub.add_parser("daemon", help="manage the daemon process")
    dsub = pd.add_subparsers(dest="daemon_command", required=True)

    p = dsub.add_parser("start", help="start the daemon and wait until ready")
    p.add_argument("--foreground", action="store_true", help="run in the foreground")
    p.add_argument("--start-timeout", type=float, default=None)
    p.add_argument("--auto-download", action="store_true",
                   help="allow the daemon to download the model (~3.5 GB) if missing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_daemon_start)

    p = dsub.add_parser("stop", help="stop the daemon")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_daemon_stop)

    p = dsub.add_parser("status", help="daemon process status (exit 3 when not running)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_daemon_status)

    p = dsub.add_parser("logs", help="print recent daemon log lines")
    p.add_argument("-n", "--lines", type=int, default=50)
    p.set_defaults(func=cmd_daemon_logs)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except CliError as exc:
        return fail(args, exc)
    except KeyboardInterrupt:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": {"code": "interrupted",
                                                     "message": "interrupted"}}))
        log_err("interrupted")
        return EXIT_INTERRUPT
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        return fail(args, CliError(EXIT_UNAVAILABLE, f"cannot reach daemon: {exc}",
                                   "daemon_unavailable"))


if __name__ == "__main__":
    sys.exit(main())
