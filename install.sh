#!/bin/sh
# Installer for the local Qwen3-TTS CLI (Apple Silicon only).
#   curl -fsSL https://raw.githubusercontent.com/season179/tts-exploration/main/install.sh | sh
# Override the ref to install with: TTS_INSTALL_REF=<tag|branch|sha>
set -eu

REF="${TTS_INSTALL_REF:-main}"
REPO="git+https://github.com/season179/tts-exploration@${REF}"
PYTHON_VERSION="3.14"  # the version this project is tested against

fail() {
    printf 'install: %s\n' "$1" >&2
    exit 1
}

[ "$(uname -s)" = "Darwin" ] || fail "macOS required (MLX runs only on Apple platforms)"
[ "$(uname -m)" = "arm64" ] || fail "Apple Silicon required (MLX does not support Intel Macs)"
# `command -v git` is not enough: macOS ships a git shim that fails until the
# Command Line Tools are actually installed.
git --version >/dev/null 2>&1 || fail "working git required — install Xcode Command Line Tools first: xcode-select --install"

UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
    echo "install: uv not found, installing it first (https://astral.sh/uv) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The current shell has not reloaded its rc files; use the absolute path.
    UV="$HOME/.local/bin/uv"
    [ -x "$UV" ] || fail "uv installed but not found at $UV; open a new shell and re-run"
fi

# A daemon left over from a previous version would keep serving old code.
if command -v tts >/dev/null 2>&1; then
    echo "install: stopping any running tts daemon from a previous version ..."
    tts daemon stop >/dev/null 2>&1 || true
fi

echo "install: installing tts from $REPO ..."
"$UV" tool install --force --python "$PYTHON_VERSION" "$REPO"

if ! command -v tts >/dev/null 2>&1; then
    echo "install: adding uv's tool directory to PATH ..."
    "$UV" tool update-shell || true
    echo "install: open a new shell (or source your shell rc) so 'tts' is found."
fi

cat <<'EOF'

Installed. Next step (one-time, ~3.5 GB model download):

    tts setup

Then:

    tts speak "hello there" -o hello.m4a
    tts --help
EOF
