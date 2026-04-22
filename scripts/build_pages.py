#!/usr/bin/env python3
"""Assemble the Pages-ready static bundle under `docs/`.

Usage:
    python scripts/build_pages.py           # regenerate + copy
    python scripts/build_pages.py --skip-sim  # copy from existing runs/demo/

This is the narrow build path the live demo ships from. It does two
jobs, in order, and nothing else:

  1. Regenerate each fixture's demo bundle under `runs/demo/<fixture>/`
     by calling `scripts/run_sim.py` and `scripts/build_atlas.py` as
     subprocesses — the same two commands `run.sh` already runs. With
     `--skip-sim`, this step is skipped and the existing on-disk
     artifacts are used as-is (useful for iterating on the landing
     page without re-simulating).

  2. Mirror the subset of `demo/` and `runs/demo/` that GitHub Pages
     needs under `docs/` — the directory Pages is configured to serve
     from. The mirror is opinionated:

       - `demo/` → `docs/demo/` (full, 3 files).
       - For each fixture, copy:
           summary.json, topology.json, atlas.json,
           audio.wav, config.yaml, atlas_audio/*.wav
         Skip state.npz (not consumed by the browser) and events.jsonl
         (empty in v0 — no event schedule fires at the control rate).

  3. Write `docs/.nojekyll` so GH Pages does not apply Jekyll rules
     (the demo uses vanilla filenames, but Jekyll would otherwise
     hide any future underscore-prefix files).

The landing page (`docs/index.html` + `docs/landing.css`) and the
existing `docs/screenshots/` are NOT touched — they are checked in
directly. This script only produces the copied / regenerated half of
the bundle.

The docs/ tree as a whole is committed: Pages on `main`/`docs` is
configured to serve it directly, no GH Actions, no gh-pages branch.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Same fixture list as run.sh. New fixtures land in both places.
FIXTURES = (
    "locked",
    "drifting",
    "two_cluster",
    "phase_beating",
    "flam",
    "polyrhythmic",
    "unstable_bridge",
    "brittle_lock",
)

# Per-fixture whitelist: the artifacts the browser reader actually
# fetches, plus config.yaml for transparency. state.npz and
# events.jsonl are deliberately dropped (not consumed; dead weight).
FIXTURE_FILES = (
    "summary.json",
    "topology.json",
    "atlas.json",
    "audio.wav",
    "config.yaml",
)
FIXTURE_DIRS = ("atlas_audio",)

DEMO_FILES = ("index.html", "app.js", "style.css")


def _run(cmd: list[str]) -> None:
    """Run a subprocess, inherit stdio; raise on non-zero exit."""
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def regenerate_fixtures(python: str) -> None:
    """Regenerate `runs/demo/<fixture>/` for every fixture."""
    runs_root = REPO / "runs" / "demo"
    runs_root.mkdir(parents=True, exist_ok=True)
    for name in FIXTURES:
        cfg = REPO / "configs" / f"regime_{name}.yaml"
        out = runs_root / name
        if not cfg.exists():
            raise SystemExit(f"error: missing {cfg}")
        print(f"generating {out} ...")
        _run([python, str(REPO / "scripts" / "run_sim.py"),
              "--config", str(cfg), "--out", str(out), "--summary-json"])
        _run([python, str(REPO / "scripts" / "build_atlas.py"),
              "--config", str(cfg), "--baseline-dir", str(out)])


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _mirror_dir(src: Path, dst: Path) -> None:
    """Replace dst with a fresh copy of src (dst is blown away first)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def assemble_bundle() -> None:
    """Copy demo/ and the runs/demo/ whitelist into docs/."""
    docs = REPO / "docs"
    docs.mkdir(exist_ok=True)

    # demo/ → docs/demo/ (exactly three files; replace wholesale so
    # stale additions in docs/demo/ cannot accumulate).
    demo_src = REPO / "demo"
    demo_dst = docs / "demo"
    demo_dst.mkdir(exist_ok=True)
    for fname in DEMO_FILES:
        src = demo_src / fname
        if not src.exists():
            raise SystemExit(f"error: missing {src}")
        _copy_file(src, demo_dst / fname)
    # Drop anything in docs/demo/ that isn't in DEMO_FILES — avoids
    # silent accumulation if the viewer is ever renamed.
    for entry in demo_dst.iterdir():
        if entry.name not in DEMO_FILES:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    # runs/demo/<fixture>/ → docs/runs/demo/<fixture>/ (whitelist).
    runs_dst_root = docs / "runs" / "demo"
    if runs_dst_root.exists():
        shutil.rmtree(runs_dst_root)
    for name in FIXTURES:
        src_dir = REPO / "runs" / "demo" / name
        dst_dir = runs_dst_root / name
        if not src_dir.exists():
            raise SystemExit(
                f"error: missing {src_dir} — run without --skip-sim "
                "or run ./run.sh once first.")
        dst_dir.mkdir(parents=True)
        for fname in FIXTURE_FILES:
            src = src_dir / fname
            if src.exists():
                _copy_file(src, dst_dir / fname)
        for dname in FIXTURE_DIRS:
            src = src_dir / dname
            if src.exists() and src.is_dir():
                _mirror_dir(src, dst_dir / dname)

    # .nojekyll — disables Jekyll processing on the served tree.
    (docs / ".nojekyll").write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="skip regenerating runs/demo/ and copy from the existing tree",
    )
    args = parser.parse_args()

    python = sys.executable
    if not args.skip_sim:
        regenerate_fixtures(python)
    assemble_bundle()

    docs = REPO / "docs"
    print()
    print(f"ok: pages bundle assembled under {docs}")
    print("  landing:  docs/index.html")
    print("  demo:     docs/demo/index.html")
    print("  runs:     docs/runs/demo/<fixture>/")
    print()
    print("preview locally:")
    print(f"  (cd {docs} && python -m http.server 8001)")
    print("  then open http://localhost:8001/")


if __name__ == "__main__":
    main()
