#!/usr/bin/env bash
# run.sh — local convenience launcher for the Resonant Instrument Lab demo.
#
# - Creates .venv if missing and installs requirements.txt into it.
# - Regenerates demo artifacts (state.npz + summary.json + topology.json
#   + events.jsonl + audio.wav + frozen config.yaml) for every fixture
#   under runs/demo/<fixture>/.
# - Prints a set of copy-pasteable browser URLs (single-run views plus
#   one compelling A/B comparison).
# - Starts a static HTTP server on PORT (default 8000, bind 127.0.0.1)
#   in the foreground. Ctrl-C stops it.
#
# Usage:
#   ./run.sh              # default port 8000
#   PORT=9000 ./run.sh    # override
#
# This is a convenience launcher, not a task runner. Scope-limited by
# design; regenerate-on-every-invoke is fine because each fixture runs
# in a few seconds and produces ~1 MB of artifacts.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

PORT="${PORT:-8000}"
FIXTURES=(locked drifting two_cluster phase_beating flam polyrhythmic unstable_bridge brittle_lock)

# --- 1. python binary discovery -------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    SYS_PY=python3
elif command -v python >/dev/null 2>&1; then
    SYS_PY=python
else
    echo "error: python3 not found on PATH." >&2
    exit 1
fi

# --- 2. venv setup --------------------------------------------------------
if [ ! -d .venv ]; then
    echo "creating .venv via $SYS_PY -m venv ..."
    "$SYS_PY" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "error: $VENV_PY missing after venv creation." >&2
    exit 1
fi

# Install requirements only if the imports fail — cheap pre-flight.
if ! "$VENV_PY" -c "import numpy, yaml" >/dev/null 2>&1; then
    echo "installing requirements.txt into .venv ..."
    "$VENV_PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt
fi

# --- 3. regenerate demo artifacts -----------------------------------------
mkdir -p runs/demo
for name in "${FIXTURES[@]}"; do
    cfg="configs/regime_$name.yaml"
    out="runs/demo/$name"
    if [ ! -f "$cfg" ]; then
        echo "error: missing $cfg" >&2
        exit 1
    fi
    echo "generating $out ..."
    "$VENV_PY" scripts/run_sim.py --config "$cfg" --out "$out" --summary-json >/dev/null
done

# --- 4. print URLs + start server -----------------------------------------
BASE="http://localhost:$PORT/demo/index.html"

printf '\n'
printf 'demo artifacts regenerated under runs/demo/.\n'
printf 'starting static server on http://localhost:%s (bind 127.0.0.1).\n\n' "$PORT"

printf 'A/B comparison — locked vs two_cluster (the dominant_cluster flip):\n'
printf '  %s?summaryA=../runs/demo/locked/summary.json&summaryB=../runs/demo/two_cluster/summary.json\n\n' "$BASE"

printf 'A/B comparison — two_cluster vs phase_beating (dominant_cluster vs drift+beating):\n'
printf '  %s?summaryA=../runs/demo/two_cluster/summary.json&summaryB=../runs/demo/phase_beating/summary.json\n\n' "$BASE"

printf 'A/B comparison — unstable_bridge single-run (counterfactual detector fires):\n'
printf '  %s?summaryA=../runs/demo/unstable_bridge/summary.json\n\n' "$BASE"

printf 'Single-run views (one per fixture):\n'
for name in "${FIXTURES[@]}"; do
    printf '  %s?summaryA=../runs/demo/%s/summary.json\n' "$BASE" "$name"
done

printf '\nCtrl-C to stop the server.\n\n'

exec "$VENV_PY" -m http.server --bind 127.0.0.1 "$PORT"
