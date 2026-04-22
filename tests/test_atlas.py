#!/usr/bin/env python3
"""Artifact-truth tests for the Intervention Atlas (scripts/build_atlas.py).

Locks in the properties that make `atlas.json` a canonical artifact rather
than a demo-shaped one:

- Generation succeeds end-to-end for the two fixtures that the README
  features as hero atlases (`regime_brittle_lock`, `regime_unstable_bridge`).
- Top-level schema shape is stable — same keys, same promised fields —
  so browser/notebook consumers can rely on it.
- Intervention count is exactly `2N` (N ablations + N nudges), matching
  the v1 bounded-enumeration contract in `scripts/build_atlas.py`.
- Byte-identical output across repeated builds of the same baseline.
  Both the JSON encoder config (`sort_keys=True`, `allow_nan=False`,
  fixed `indent`, trailing newline) and the upstream sim determinism
  are load-bearing here — the check catches drift in either.
- Two grounded semantic assertions — narrow, robust, picked from the
  fixture stories documented in the YAML configs:
    * `regime_unstable_bridge` ranks `ablate_n0` (the bridge) at or near
      the top, because removing the load-bearing bridge collapses cluster A.
    * `regime_brittle_lock` places at least one intervention on a
      brittle node (ω-outer nodes 2/3/4) in its top-3, because the
      fixture's brittleness story lives on exactly those nodes.

Run directly:
    python tests/test_atlas.py
Or via pytest:
    python -m pytest tests/test_atlas.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import load  # noqa: E402
from scripts.build_atlas import (  # noqa: E402
    ATLAS_SCHEMA_VERSION,
    OBSERVATIONAL_DETECTORS,
    _build_atlas,
    _write_atlas,
)
from scripts.run_sim import _build_summary, _write_summary_json  # noqa: E402
from sim.garden import simulate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO / "configs"

# Top-level keys the atlas schema promises. Readers (demo/app.js, future
# notebooks) key on exactly these — drift here is a breaking change and
# belongs in a schema_version bump, not silent reshape.
REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "baseline",
    "intervention_families",
    "observational_detectors",
    "ranking",
    "interventions",
}

# Keys required inside each intervention entry. Same contract note.
REQUIRED_INTERVENTION_KEYS = {
    "id",
    "kind",
    "node",
    "delta_hz",
    "label",
    "summary",
    "audio_path",
    "deltas",
    "score",
    "score_components",
    "score_explanation",
}


def _materialize_baseline(tmp_root: Path, fixture_name: str) -> tuple[dict, Path, Path]:
    """Run the baseline sim + summary for `fixture_name` into `tmp_root`.

    Returns `(cfg, config_path, baseline_dir)`. The baseline dir holds
    `state.npz` and `summary.json`, which is all `_build_atlas` needs.
    We rebuild from the YAML config rather than reusing `runs/demo/` so
    the test does not require the convenience script to have been run,
    and so the repo's checked-in `runs/demo/` can move without breaking
    the test.
    """
    config_path = CONFIGS_DIR / f"regime_{fixture_name}.yaml"
    cfg = load(config_path)
    baseline_dir = tmp_root / "baseline"
    simulate(cfg, baseline_dir, config_path=str(config_path))
    summary = _build_summary(cfg, str(config_path), baseline_dir)
    _write_summary_json(summary, baseline_dir)
    return cfg, config_path, baseline_dir


def _validate_schema_shape(atlas: dict, N: int) -> None:
    """Assert the top-level atlas dict carries the documented schema."""
    missing = REQUIRED_TOP_LEVEL_KEYS - set(atlas.keys())
    assert not missing, f"atlas.json missing top-level keys: {sorted(missing)}"

    assert atlas["schema_version"] == ATLAS_SCHEMA_VERSION, (
        f"schema_version mismatch: {atlas['schema_version']} != {ATLAS_SCHEMA_VERSION}"
    )
    assert atlas["intervention_families"] == ["ablate_node", "nudge_node"], (
        f"intervention_families drifted: {atlas['intervention_families']}"
    )
    assert atlas["observational_detectors"] == list(OBSERVATIONAL_DETECTORS), (
        "observational_detectors list drifted from build_atlas.OBSERVATIONAL_DETECTORS"
    )

    baseline = atlas["baseline"]
    for key in ("config_path", "N", "summary_path", "observational_summary"):
        assert key in baseline, f"baseline block missing '{key}'"
    assert baseline["N"] == N, f"baseline.N mismatch: {baseline['N']} != {N}"

    ranking = atlas["ranking"]
    assert "score_formula" in ranking, "ranking block must document score_formula"

    assert isinstance(atlas["interventions"], list)
    # Each intervention entry carries the full documented block.
    for i, iv in enumerate(atlas["interventions"]):
        missing_iv = REQUIRED_INTERVENTION_KEYS - set(iv.keys())
        assert not missing_iv, (
            f"intervention[{i}] ({iv.get('id')}) missing keys: {sorted(missing_iv)}"
        )
        assert iv["kind"] in ("ablate_node", "nudge_node"), (
            f"intervention[{i}] kind '{iv['kind']}' not in v1 families"
        )
        assert 0 <= iv["node"] < N, (
            f"intervention[{i}] node {iv['node']} out of range [0,{N})"
        )


def _assert_ranked_descending(atlas: dict) -> None:
    scores = [iv["score"] for iv in atlas["interventions"]]
    assert scores == sorted(scores, reverse=True), (
        "interventions must be ranked by score descending"
    )


# ---------------------------------------------------------------------------
# Smoke + schema + count — run once per fixture, assert the full contract.

def _smoke_one_fixture(fixture_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        cfg, config_path, baseline_dir = _materialize_baseline(tmp_root, fixture_name)
        N = int(cfg["scene"]["N"])

        atlas = _build_atlas(cfg, str(config_path), baseline_dir)

        _validate_schema_shape(atlas, N)
        _assert_ranked_descending(atlas)

        # 2N: N ablations + N nudges, no more, no less.
        assert len(atlas["interventions"]) == 2 * N, (
            f"{fixture_name}: expected 2N={2*N} interventions, "
            f"got {len(atlas['interventions'])}"
        )
        n_ablate = sum(1 for iv in atlas["interventions"] if iv["kind"] == "ablate_node")
        n_nudge = sum(1 for iv in atlas["interventions"] if iv["kind"] == "nudge_node")
        assert n_ablate == N and n_nudge == N, (
            f"{fixture_name}: expected N ablations + N nudges, got "
            f"ablate={n_ablate}, nudge={n_nudge}"
        )

        # Node coverage: each node is hit by exactly one ablation and one nudge.
        ablated_nodes = sorted(
            iv["node"] for iv in atlas["interventions"] if iv["kind"] == "ablate_node"
        )
        nudged_nodes = sorted(
            iv["node"] for iv in atlas["interventions"] if iv["kind"] == "nudge_node"
        )
        assert ablated_nodes == list(range(N))
        assert nudged_nodes == list(range(N))

        # Atlas actually writes to disk with the same payload shape.
        out_path = _write_atlas(atlas, baseline_dir)
        assert out_path.is_file(), f"{out_path} not written"
        roundtrip = json.loads(out_path.read_text())
        _validate_schema_shape(roundtrip, N)
        assert len(roundtrip["interventions"]) == 2 * N

        print(
            f"ok: atlas smoke — {fixture_name}: N={N}, "
            f"{len(atlas['interventions'])} interventions, "
            f"top={atlas['interventions'][0]['id']} "
            f"(score={atlas['interventions'][0]['score']:.3f})"
        )


def test_atlas_smoke_brittle_lock():
    _smoke_one_fixture("brittle_lock")


def test_atlas_smoke_unstable_bridge():
    _smoke_one_fixture("unstable_bridge")


# ---------------------------------------------------------------------------
# Determinism — byte-identical atlas.json across two builds of one fixture.

def test_atlas_determinism_byte_identical():
    """Same (config + baseline) twice → byte-identical atlas.json.

    This is a strong deterministic contract: it catches regressions in
    both the upstream sim (state.npz bit-for-bit stable) and the atlas
    writer (`sort_keys=True`, `allow_nan=False`, fixed indent, trailing
    newline). If this assertion ever weakens, bump ATLAS_SCHEMA_VERSION
    and document the weaker contract in `scripts/build_atlas.py`.
    """
    fixture = "brittle_lock"  # N=5 keeps the test cheap (~10 sims total)
    config_path = CONFIGS_DIR / f"regime_{fixture}.yaml"
    cfg = load(config_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        # One baseline, reused by both atlas builds — atlas.json is the
        # only thing under test. Copy, don't rerun, so a regression in
        # atlas-side determinism is not masked by sim-side redetermined
        # identical state.npz (that's covered by test_ablate's
        # determinism tests and re-covered below via the a/b paths).
        cfg_src, _, baseline_dir = _materialize_baseline(tmp_root, fixture)
        del cfg_src  # loaded above for test independence

        dir_a = tmp_root / "a"
        dir_b = tmp_root / "b"
        shutil.copytree(baseline_dir, dir_a)
        shutil.copytree(baseline_dir, dir_b)

        atlas_a = _build_atlas(cfg, str(config_path), dir_a)
        atlas_b = _build_atlas(cfg, str(config_path), dir_b)
        path_a = _write_atlas(atlas_a, dir_a)
        path_b = _write_atlas(atlas_b, dir_b)

        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()
        assert bytes_a == bytes_b, (
            f"atlas.json not byte-identical across two builds "
            f"(len_a={len(bytes_a)}, len_b={len(bytes_b)})"
        )
        print(
            f"ok: atlas determinism — {fixture}: {len(bytes_a)} bytes, "
            f"byte-identical across two builds"
        )


# ---------------------------------------------------------------------------
# Grounded semantic assertions — narrow, anchored to fixture stories.

def test_unstable_bridge_ranks_ablate_n0_near_top():
    """`regime_unstable_bridge`: removing the bridge (node 0) must land
    in the top of the ranking.

    The fixture story (configs/regime_unstable_bridge.yaml) is that node
    0 is the one load-bearing spoke between the two far-apart spokes
    1 and 2 — ablating it collapses cluster A's coherence. The atlas
    should therefore rank `ablate_n0` as one of the highest-impact
    interventions. We assert "in the top 3" rather than "exactly #1"
    because nudges to the same bridge/spoke can score in the same
    high band, and ordering between them is not the load-bearing claim.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        cfg, config_path, baseline_dir = _materialize_baseline(tmp_root, "unstable_bridge")
        atlas = _build_atlas(cfg, str(config_path), baseline_dir)

        top_ids = [iv["id"] for iv in atlas["interventions"][:3]]
        assert "ablate_n0" in top_ids, (
            "unstable_bridge atlas should rank `ablate_n0` (the bridge) in the "
            f"top 3. Got top 3 = {top_ids}"
        )
        # ablate_n0 must at minimum have caused a detector flip — the
        # bridge ablation collapses dominant_cluster in the fixture's story.
        entry = next(iv for iv in atlas["interventions"] if iv["id"] == "ablate_n0")
        assert entry["score_components"]["flips"] >= 1, (
            f"ablate_n0 should flip at least one detector vs baseline; "
            f"got flips={entry['score_components']['flips']}"
        )
        print(
            f"ok: unstable_bridge semantic — ablate_n0 in top 3 (top={top_ids[0]}, "
            f"ablate_n0 flips={entry['score_components']['flips']})"
        )


def test_brittle_lock_top_intervention_on_brittle_node():
    """`regime_brittle_lock`: at least one of the top-ranked interventions
    must land on a node the baseline detector flagged as brittle.

    The fixture story (configs/regime_brittle_lock.yaml) is that the
    outer-ω nodes (2, 3, 4) are the ones whose +0.25 Hz nudge breaks
    the lock — `detect_brittle_lock` surfaces those as `brittle_nodes`
    in the baseline summary. A credible atlas ranking places at least
    one ablation or nudge on those nodes near the top, because taking
    them out (or detuning them) is what actually breaks the lock.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        cfg, config_path, baseline_dir = _materialize_baseline(tmp_root, "brittle_lock")

        with (baseline_dir / "summary.json").open() as f:
            baseline_summary = json.load(f)
        brittle_nodes = set(
            baseline_summary["detectors"]["brittle_lock"].get("brittle_nodes") or []
        )
        assert brittle_nodes, (
            "baseline brittle_lock should carry non-empty brittle_nodes "
            "(fixture precondition for this assertion)"
        )

        atlas = _build_atlas(cfg, str(config_path), baseline_dir)
        top3 = atlas["interventions"][:3]
        hits = [iv for iv in top3 if iv["node"] in brittle_nodes]
        assert hits, (
            f"no top-3 intervention lands on a brittle node "
            f"(brittle_nodes={sorted(brittle_nodes)}, "
            f"top3={[(iv['id'], iv['node']) for iv in top3]})"
        )
        print(
            f"ok: brittle_lock semantic — brittle_nodes={sorted(brittle_nodes)}, "
            f"top-3 hits on brittle nodes = "
            f"{[iv['id'] for iv in hits]}"
        )


# ---------------------------------------------------------------------------
# Audio artifacts — each intervention must carry a playable WAV on disk.

def test_atlas_audio_artifacts_exist_per_intervention():
    """Every intervention's `audio_path` must point at a real, non-empty
    WAV file that the browser A/B player can load.

    Contract: `_write_atlas` writes atlas.json alongside a sibling
    `atlas_audio/<id>.wav` for each intervention, and records the
    relative path on each entry. This test catches:
      - missing or misnamed audio files on disk,
      - `audio_path` drift in the schema,
      - the 0-byte silent-wav regression (peak amplitude must be > 0 —
        the pulse synthesis should actually render sound, not just
        format-correct silence).
    """
    fixture = "brittle_lock"  # N=5 keeps the sim bundle cheap
    config_path = CONFIGS_DIR / f"regime_{fixture}.yaml"
    cfg = load(config_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        cfg_src, _, baseline_dir = _materialize_baseline(tmp_root, fixture)
        del cfg_src
        atlas = _build_atlas(cfg, str(config_path), baseline_dir)
        _write_atlas(atlas, baseline_dir)

        # Baseline's own audio.wav must be referenced and audible.
        assert atlas["baseline"]["audio_path"] == "audio.wav", (
            f"baseline.audio_path drifted: {atlas['baseline']['audio_path']!r}"
        )
        baseline_wav = baseline_dir / "audio.wav"
        assert baseline_wav.is_file(), f"baseline audio missing at {baseline_wav}"
        _assert_wav_has_signal(baseline_wav)

        for iv in atlas["interventions"]:
            rel = iv["audio_path"]
            assert isinstance(rel, str) and rel, (
                f"{iv['id']}: audio_path must be a non-empty string, got {rel!r}"
            )
            assert rel.startswith("atlas_audio/") and rel.endswith(".wav"), (
                f"{iv['id']}: audio_path shape drifted — got {rel!r}"
            )
            wav = baseline_dir / rel
            assert wav.is_file(), (
                f"{iv['id']}: audio_path points to missing file {wav}"
            )
            _assert_wav_has_signal(wav)

        print(
            f"ok: atlas audio — {fixture}: baseline + "
            f"{len(atlas['interventions'])} intervention WAVs written and audible"
        )


def _assert_wav_has_signal(path):
    """Peak absolute sample > 0 — catches the silent-WAV regression."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    arr = np.frombuffer(frames, dtype=np.int16)
    peak = int(np.abs(arr).max()) if arr.size else 0
    assert peak > 0, f"{path} is silent (peak sample = 0)"


if __name__ == "__main__":
    test_atlas_smoke_brittle_lock()
    test_atlas_smoke_unstable_bridge()
    test_atlas_determinism_byte_identical()
    test_unstable_bridge_ranks_ablate_n0_near_top()
    test_brittle_lock_top_intervention_on_brittle_node()
    test_atlas_audio_artifacts_exist_per_intervention()
