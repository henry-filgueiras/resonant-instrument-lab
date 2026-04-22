#!/usr/bin/env python3
"""Detector-level tests.

Exercise the semantic detectors in `sim.detectors` against the three
regime fixtures (`locked`, `drifting`, `two_cluster`). Each detector
must fire on its namesake regime and stay silent on the other two.

These are not simulator smoke tests (those live in `test_regimes.py`) —
the simulator is trusted here; what we're checking is that the detector
thresholds and sustained-window settings correctly discriminate the
regimes that the fixtures already validate as physically distinct.

Run directly:
    python tests/test_detectors.py
Or via pytest:
    python -m pytest tests/
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim import load, simulate  # noqa: E402
from sim.detectors import (  # noqa: E402
    BRITTLE_LOCK_MIN_R,
    BRITTLE_LOCK_NUDGE_HZ,
    BRITTLE_LOCK_TAIL_S,
    DOMINANT_CLUSTER_MIN_LOCAL_R,
    DOMINANT_CLUSTER_MIN_SEP,
    DOMINANT_CLUSTER_MIN_SIZE,
    DOMINANT_CLUSTER_TAIL_S,
    DRIFTING_MIN_S,
    FLAM_DW_MAX,
    FLAM_MAX_OFFSET_S,
    FLAM_MIN_EVENTS,
    FLAM_MIN_OFFSET_S,
    FLAM_WARMUP_S,
    PHASE_BEATING_DW_MAX,
    PHASE_BEATING_DW_MIN,
    PHASE_BEATING_MIN_R2,
    PHASE_BEATING_WARMUP_S,
    PHASE_LOCKED_MIN_S,
    POLY_MIN_PULSES_PER_NODE,
    POLY_NEAR_UNITY_MAX,
    POLY_RATIO_SET,
    POLY_RATIO_TOL,
    POLY_SUB_WINDOW_RATIO_TOL,
    POLY_WARMUP_S,
    UNSTABLE_BRIDGE_MIN_CLUSTER_SIZE,
    UNSTABLE_BRIDGE_TAIL_S,
    _brittle_lock_evidence,
    _dominant_cluster_evidence,
    _flam_pair_evidence,
    _phase_beating_pair_evidence,
    _polyrhythmic_pair_evidence,
    _unstable_bridge_evidence,
    detect_brittle_lock,
    detect_dominant_cluster,
    detect_drifting,
    detect_flam,
    detect_phase_beating,
    detect_phase_locked,
    detect_polyrhythmic,
    detect_unstable_bridge,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _run(config_path):
    cfg = load(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        simulate(cfg, tmp, config_path=str(config_path))
        with np.load(Path(tmp) / "state.npz") as state:
            artifacts = {k: state[k] for k in state.files}
    return cfg, artifacts


def _assert_silent(result, detector_name, regime_name):
    assert not result.fired, (
        f"{detector_name} should not fire on {regime_name}, but did: "
        f"{len(result.windows)} window(s), longest = {result.longest_window_frames} frames"
    )
    assert result.windows == [], f"{detector_name} on {regime_name}: expected no windows"
    assert result.confidence == 0.0, (
        f"{detector_name} on {regime_name}: expected confidence 0.0 for silent detector, "
        f"got {result.confidence}"
    )
    assert result.longest_window_frames == 0


# ---------------------------------------------------------------------------
# phase_locked — fires only on the locked regime.

def test_detect_phase_locked_across_regimes():
    # (a) fires cleanly on locked
    cfg, art = _run(CONFIGS_DIR / "regime_locked.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_phase_locked(art["theta"], rate)
    assert result.fired, "phase_locked should fire on regime_locked"
    assert result.longest_window_frames >= int(PHASE_LOCKED_MIN_S * rate), (
        f"expected a >= {PHASE_LOCKED_MIN_S}s lock window, got "
        f"{result.longest_window_frames / rate:.2f}s"
    )
    assert result.confidence > 0.05, (
        f"expected solid margin above R_LOCK (observed r ≈ 0.995 → margin ≈ 0.09), "
        f"got confidence {result.confidence:.4f}"
    )

    # (b) silent on drifting — r never reaches R_LOCK
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_phase_locked(art_d["theta"], cfg_d["run"]["control_rate_hz"]),
        "phase_locked", "regime_drifting",
    )

    # (c) silent on two_cluster — r spikes above R_LOCK but never sustains the window
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_phase_locked(art_t["theta"], cfg_t["run"]["control_rate_hz"]),
        "phase_locked", "regime_two_cluster",
    )

    print(
        f"ok: phase_locked — locked fired (conf={result.confidence:.4f}, "
        f"longest={result.longest_window_frames/rate:.2f}s); drifting & two_cluster silent"
    )


# ---------------------------------------------------------------------------
# drifting — fires only on the drifting regime.

def test_detect_drifting_across_regimes():
    # (a) fires cleanly on drifting
    cfg, art = _run(CONFIGS_DIR / "regime_drifting.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_drifting(art["theta"], rate)
    assert result.fired, "drifting should fire on regime_drifting"
    assert result.longest_window_frames >= int(DRIFTING_MIN_S * rate), (
        f"expected a >= {DRIFTING_MIN_S}s low-coherence window, got "
        f"{result.longest_window_frames / rate:.2f}s"
    )
    assert result.confidence > 0.10, (
        f"expected mean r well below R_DRIFT (observed r ≈ 0.32 → margin ≈ 0.18), "
        f"got confidence {result.confidence:.4f}"
    )

    # (b) silent on locked — r is pinned near 1.0
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_drifting(art_l["theta"], cfg_l["run"]["control_rate_hz"]),
        "drifting", "regime_locked",
    )

    # (c) silent on two_cluster — low-r valleys stay below the 2.0s window minimum
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_drifting(art_t["theta"], cfg_t["run"]["control_rate_hz"]),
        "drifting", "regime_two_cluster",
    )

    print(
        f"ok: drifting — drifting fired (conf={result.confidence:.4f}, "
        f"longest={result.longest_window_frames/rate:.2f}s); locked & two_cluster silent"
    )


# ---------------------------------------------------------------------------
# phase_beating — fires only on the dedicated beating fixture.
#
# The three original regime fixtures are all negatives for this detector:
#   - locked: every pair sits at |Δω̄| ≈ 0, below the lower velocity gate.
#   - drifting: the smallest candidate pair has |Δω̄| ≈ 1.59 rad/s, above
#     the 1.5 rad/s "close in mean phase velocity" upper gate — the
#     drifting regime has genuinely structured pairwise drift, but the
#     pairs are no longer near-frequency by this detector's definition.
#   - two_cluster: intra-cluster pairs are locked (|Δω̄| ≈ 0), inter-
#     cluster pairs sit at ~5.6 rad/s (above the upper gate).
# `regime_phase_beating` is the single positive fixture. Its pair (0,1)
# sits at |Δω̄| ≈ 1.22 rad/s with beat period ~5.1 s; the 12 s run covers
# ~2 full cycles of the 2π coverage requirement.

def test_detect_phase_beating_across_regimes():
    # (a) fires on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_phase_beating(
        art["theta"], art["phase_vel"], art["pulse_fired"], rate
    )
    assert result.fired, "phase_beating should fire on regime_phase_beating"
    # Analysis window is the run minus the warmup skip.
    expected_window_frames = art["theta"].shape[0] - int(PHASE_BEATING_WARMUP_S * rate)
    assert result.longest_window_frames == expected_window_frames
    assert result.windows == [(int(PHASE_BEATING_WARMUP_S * rate), art["theta"].shape[0])]
    # R² margin above the 0.9 floor; the fixture's pair sits near R² ≈ 0.998.
    assert result.confidence > 0.05, (
        f"expected R² margin > 0.05 above floor {PHASE_BEATING_MIN_R2}, "
        f"got confidence {result.confidence:.4f}"
    )

    # Confirm exactly the (0, 1) pair contributed — the distractor nodes
    # sit too far away in frequency to enter the candidate band.
    evidence = _phase_beating_pair_evidence(
        art["theta"], art["phase_vel"], art["pulse_fired"], rate
    )
    assert len(evidence) == 1, f"expected one qualifying pair, got {len(evidence)}: {evidence}"
    e = evidence[0]
    assert (e["i"], e["j"]) == (0, 1)
    assert PHASE_BEATING_DW_MIN <= abs(e["dw"]) <= PHASE_BEATING_DW_MAX
    assert e["r2"] >= PHASE_BEATING_MIN_R2

    # (b) silent on locked — every pair has Δω̄ ≈ 0, below lower gate
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_phase_beating(
            art_l["theta"], art_l["phase_vel"], art_l["pulse_fired"],
            cfg_l["run"]["control_rate_hz"],
        ),
        "phase_beating", "regime_locked",
    )

    # (c) silent on drifting — smallest candidate |Δω̄| ≈ 1.59 rad/s is
    # above the upper gate; the drifting regime has structured pairwise
    # drift but its pairs are not "close in mean phase velocity" by this
    # detector's threshold policy.
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_phase_beating(
            art_d["theta"], art_d["phase_vel"], art_d["pulse_fired"],
            cfg_d["run"]["control_rate_hz"],
        ),
        "phase_beating", "regime_drifting",
    )

    # (d) silent on two_cluster — intra-cluster pairs locked (|Δω̄| ≈ 0),
    # inter-cluster pairs at ~5.6 rad/s (above upper gate).
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_phase_beating(
            art_t["theta"], art_t["phase_vel"], art_t["pulse_fired"],
            cfg_t["run"]["control_rate_hz"],
        ),
        "phase_beating", "regime_two_cluster",
    )

    print(
        f"ok: phase_beating — phase_beating fixture fired "
        f"(conf={result.confidence:.4f}, |Δω̄|={abs(e['dw']):.3f} rad/s, "
        f"beat~{(2 * np.pi) / abs(e['dw']):.2f} s, R²={e['r2']:.4f}); "
        f"locked, drifting, two_cluster silent"
    )


# ---------------------------------------------------------------------------
# flam — fires only on the dedicated flam fixture.
#
# The four earlier regime fixtures are all negatives for this detector:
#   - regime_locked: the extreme pair in an 8-node lock produces ~20 ms
#     offsets (inside the flam band) but intermediate nodes fire between
#     them, caught by the isolation margin.
#   - regime_drifting: every candidate pair has |Δω̄| > 1.5 rad/s,
#     above the 0.25 rad/s velocity gate.
#   - regime_two_cluster: the leading intra-cluster pair fires 15 ms
#     apart with the right velocity match, but the rest of the cluster
#     fires 5–10 ms just outside the pair window, caught by the
#     isolation margin.
#   - regime_phase_beating: the beating pair sits at |Δω̄| ≈ 1.22 rad/s,
#     above the velocity gate; it drifts through all offsets rather
#     than holding one.
# `regime_flam` is the single positive fixture: two nodes at 2.00 /
# 2.20 Hz locked together with a ~33 ms phase offset, two distractors
# at 4 / 5 Hz placed far away.

def test_detect_flam_across_regimes():
    # (a) fires cleanly on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_flam.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_flam(art["phase_vel"], art["pulse_fired"], rate)
    assert result.fired, "flam should fire on regime_flam"
    expected_window_frames = art["pulse_fired"].shape[0] - int(FLAM_WARMUP_S * rate)
    assert result.longest_window_frames == expected_window_frames
    assert result.windows == [
        (int(FLAM_WARMUP_S * rate), art["pulse_fired"].shape[0])
    ]
    # 11 qualifying events vs 6 minimum → margin ≈ (11-6)/6 ≈ 0.83.
    assert result.confidence > 0.3, (
        f"expected a comfortable margin over {FLAM_MIN_EVENTS}-event floor, "
        f"got confidence {result.confidence:.4f}"
    )

    # Exactly the (0, 1) pair contributes; offset is in band and consistent.
    evidence = _flam_pair_evidence(art["phase_vel"], art["pulse_fired"], rate)
    assert len(evidence) == 1, f"expected one qualifying pair, got {len(evidence)}: {evidence}"
    e = evidence[0]
    assert (e["i"], e["j"]) == (0, 1)
    assert abs(e["dw"]) <= FLAM_DW_MAX
    assert FLAM_MIN_OFFSET_S <= e["median_abs_offset_s"] <= FLAM_MAX_OFFSET_S
    assert e["n_events"] >= FLAM_MIN_EVENTS

    # (b) silent on locked — extreme pair has crowd members inside margin
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_flam(art_l["phase_vel"], art_l["pulse_fired"],
                    cfg_l["run"]["control_rate_hz"]),
        "flam", "regime_locked",
    )

    # (c) silent on drifting — all candidate pairs above the velocity gate
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_flam(art_d["phase_vel"], art_d["pulse_fired"],
                    cfg_d["run"]["control_rate_hz"]),
        "flam", "regime_drifting",
    )

    # (d) silent on two_cluster — intra-cluster leading pair has neighbors
    # firing just outside the pair window, inside the isolation margin
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_flam(art_t["phase_vel"], art_t["pulse_fired"],
                    cfg_t["run"]["control_rate_hz"]),
        "flam", "regime_two_cluster",
    )

    # (e) silent on phase_beating — beating pair |Δω̄| > velocity gate
    cfg_b, art_b = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    _assert_silent(
        detect_flam(art_b["phase_vel"], art_b["pulse_fired"],
                    cfg_b["run"]["control_rate_hz"]),
        "flam", "regime_phase_beating",
    )

    print(
        f"ok: flam — flam fixture fired "
        f"(conf={result.confidence:.4f}, |Δω̄|={abs(e['dw']):.3f} rad/s, "
        f"median offset={e['median_abs_offset_s']*1000:.1f} ms, "
        f"n_events={e['n_events']}); locked, drifting, two_cluster, "
        f"phase_beating silent"
    )


# ---------------------------------------------------------------------------
# dominant_cluster — fires only on the two_cluster fixture.
#
# The other four fixtures are all negatives under the detector's gates:
#   - regime_locked: all mean velocities collapse onto one value →
#     separability_ratio = 0 (no gap), below the 5.0 gate. Even though
#     local_r = 1.0 for any subset, there is no genuine two-way split.
#   - regime_drifting: sorted velocities form a ramp with no dominant
#     gap → separability ≈ 1.4, below the 5.0 gate; local r in both
#     candidate groups sits around 0.4–0.6, well below the 0.9 floor.
#   - regime_phase_beating: separability ≈ 1.8 (below gate); no group
#     reaches the 0.9 coherence floor (candidates sit at 0.33–0.58).
#   - regime_flam: at N=4 the pair is internally coherent (local_r ≈
#     0.98) but separability ≈ 1.9, below the gate — the coherent pair
#     does not stand cleanly apart from the two distractors.
# regime_two_cluster is the single positive: separability ≈ 706, clean
# 4+4 split, both sub-clusters locked at local_r ≈ 0.993.

def test_detect_dominant_cluster_across_regimes():
    # (a) fires cleanly on two_cluster
    cfg, art = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_dominant_cluster(art["theta"], art["phase_vel"], rate)
    assert result.fired, "dominant_cluster should fire on regime_two_cluster"
    tail_frames = int(DOMINANT_CLUSTER_TAIL_S * rate)
    assert result.longest_window_frames == tail_frames
    T = art["theta"].shape[0]
    assert result.windows == [(T - tail_frames, T)]
    # Both sub-clusters at local_r ≈ 0.993 → margin ≈ 0.09 above 0.9 floor.
    assert result.confidence > 0.05, (
        f"expected solid margin above local_r floor {DOMINANT_CLUSTER_MIN_LOCAL_R}, "
        f"got confidence {result.confidence:.4f}"
    )

    # Both clusters contribute, both above the coherence floor, sizes 4+4,
    # separability far above the gate.
    evidence = _dominant_cluster_evidence(art["theta"], art["phase_vel"], rate)
    assert len(evidence) == 2, f"expected both clusters to qualify, got {len(evidence)}: {evidence}"
    for e in evidence:
        assert e["size"] >= DOMINANT_CLUSTER_MIN_SIZE
        assert e["local_r"] >= DOMINANT_CLUSTER_MIN_LOCAL_R
        assert e["separability_ratio"] >= DOMINANT_CLUSTER_MIN_SEP
    # Clean 4+4 split expected for this fixture.
    sizes = sorted(e["size"] for e in evidence)
    assert sizes == [4, 4], f"expected balanced 4+4 split, got {sizes}"

    # (b) silent on locked — separability = 0 (all velocities equal)
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_dominant_cluster(
            art_l["theta"], art_l["phase_vel"], cfg_l["run"]["control_rate_hz"]
        ),
        "dominant_cluster", "regime_locked",
    )

    # (c) silent on drifting — separability ≈ 1.4 (below gate), no
    # candidate sub-cluster is internally coherent
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_dominant_cluster(
            art_d["theta"], art_d["phase_vel"], cfg_d["run"]["control_rate_hz"]
        ),
        "dominant_cluster", "regime_drifting",
    )

    # (d) silent on phase_beating — separability ≈ 1.8 (below gate); no
    # sub-cluster reaches the 0.9 coherence floor
    cfg_b, art_b = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    _assert_silent(
        detect_dominant_cluster(
            art_b["theta"], art_b["phase_vel"], cfg_b["run"]["control_rate_hz"]
        ),
        "dominant_cluster", "regime_phase_beating",
    )

    # (e) silent on flam — the coherent pair exists but separability ≈
    # 1.9 (below the 5.0 gate); the pair does not stand cleanly apart
    # from the distractors in the 3 s tail
    cfg_f, art_f = _run(CONFIGS_DIR / "regime_flam.yaml")
    _assert_silent(
        detect_dominant_cluster(
            art_f["theta"], art_f["phase_vel"], cfg_f["run"]["control_rate_hz"]
        ),
        "dominant_cluster", "regime_flam",
    )

    sep = evidence[0]["separability_ratio"]
    locals_r = sorted(e["local_r"] for e in evidence)
    print(
        f"ok: dominant_cluster — two_cluster fired "
        f"(conf={result.confidence:.4f}, sizes={sorted(e['size'] for e in evidence)}, "
        f"local_r={[f'{x:.3f}' for x in locals_r]}, sep={sep:.1f}); "
        f"locked, drifting, phase_beating, flam silent"
    )


# ---------------------------------------------------------------------------
# polyrhythmic — fires only on the dedicated polyrhythmic fixture.
#
# The five earlier regime fixtures are all negatives for this detector:
#   - regime_locked: all nodes frequency-lock at a common mean rate, so
#     every pair sits near 1:1 — excluded by POLY_NEAR_UNITY_MAX.
#   - regime_drifting: the near-misses to 2:3 land at 2.7 % / 3.2 %
#     relative deviation — above the 2 % whole-window tolerance. The
#     2.10/2.60 Hz pair sits within ~1 % of 4:5, which is exactly why
#     4:5 is not in POLY_RATIO_SET for this pass.
#   - regime_two_cluster: inter-cluster ratio ≈ 2.11/3.01 = 0.701,
#     closest to 2:3 at ~5 %.
#   - regime_phase_beating: the beating pair sits near 1:1 (excluded);
#     distractors were retuned this pass (6.0 → 6.3 Hz) to break the
#     two exact integer ratios (4.0/6.0 = 2:3 and 4.5/6.0 = 3:4) the
#     old set carried.
#   - regime_flam: the flam pair sits near 1:1 (excluded); the distractor
#     pair matches 4:5, which is not in the detector's set.
# regime_polyrhythmic is the single positive fixture: two nodes at 2.0
# and 3.0 Hz with coupling weak enough that each runs at its intrinsic
# rate, preserving a clean 2:3 pulse-rate ratio.

def test_detect_polyrhythmic_across_regimes():
    # (a) fires cleanly on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_polyrhythmic.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_polyrhythmic(art["pulse_fired"], rate)
    assert result.fired, "polyrhythmic should fire on regime_polyrhythmic"
    expected_window_frames = art["pulse_fired"].shape[0] - int(POLY_WARMUP_S * rate)
    assert result.longest_window_frames == expected_window_frames
    assert result.windows == [
        (int(POLY_WARMUP_S * rate), art["pulse_fired"].shape[0])
    ]
    # Deviation well under 2 % → margin near the top of the [0, 0.02] band.
    assert result.confidence > 0.5 * POLY_RATIO_TOL, (
        f"expected ratio-margin more than half the {POLY_RATIO_TOL} tolerance, "
        f"got confidence {result.confidence:.4f}"
    )

    # Exactly the (0, 1) pair contributes, locked to 2:3, with every
    # sub-window inside the stricter sub-window tolerance.
    evidence = _polyrhythmic_pair_evidence(art["pulse_fired"], rate)
    assert len(evidence) == 1, f"expected one qualifying pair, got {len(evidence)}: {evidence}"
    e = evidence[0]
    assert (e["i"], e["j"]) == (0, 1)
    assert (e["p"], e["q"]) == (2, 3)
    assert (2, 3) in POLY_RATIO_SET
    assert e["ratio_dev"] <= POLY_RATIO_TOL
    assert e["n_pulses_slow"] >= POLY_MIN_PULSES_PER_NODE
    assert e["n_pulses_fast"] >= POLY_MIN_PULSES_PER_NODE
    assert e["rate_slow"] < e["rate_fast"]
    # Slow rate ≈ 2 Hz, fast ≈ 3 Hz (intrinsic, within the weak-coupling shift).
    assert abs(e["rate_slow"] - 2.0) < 0.05
    assert abs(e["rate_fast"] - 3.0) < 0.05
    assert len(e["sub_window_devs"]) == 3
    for dev in e["sub_window_devs"]:
        assert dev <= POLY_SUB_WINDOW_RATIO_TOL
    # Pair rate ratio is below the near-unity gate (so it's a legitimate
    # small-integer claim, not a phase_beating / flam / lock near-miss).
    assert e["ratio"] <= POLY_NEAR_UNITY_MAX

    # (b) silent on locked — every pair sits near 1:1
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_polyrhythmic(art_l["pulse_fired"], cfg_l["run"]["control_rate_hz"]),
        "polyrhythmic", "regime_locked",
    )

    # (c) silent on drifting — near-misses to 2:3 sit at >2 % relative
    # deviation; the 4:5 near-miss is rejected because 4:5 is not in
    # POLY_RATIO_SET this pass
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_polyrhythmic(art_d["pulse_fired"], cfg_d["run"]["control_rate_hz"]),
        "polyrhythmic", "regime_drifting",
    )

    # (d) silent on two_cluster — inter-cluster ratio ~5 % above 2:3
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_polyrhythmic(art_t["pulse_fired"], cfg_t["run"]["control_rate_hz"]),
        "polyrhythmic", "regime_two_cluster",
    )

    # (e) silent on phase_beating — retuned distractors leave no exact
    # 2:3 or 3:4 pair
    cfg_b, art_b = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    _assert_silent(
        detect_polyrhythmic(art_b["pulse_fired"], cfg_b["run"]["control_rate_hz"]),
        "polyrhythmic", "regime_phase_beating",
    )

    # (f) silent on flam — distractor pair matches 4:5, which is not in
    # the detector's ratio set
    cfg_f, art_f = _run(CONFIGS_DIR / "regime_flam.yaml")
    _assert_silent(
        detect_polyrhythmic(art_f["pulse_fired"], cfg_f["run"]["control_rate_hz"]),
        "polyrhythmic", "regime_flam",
    )

    print(
        f"ok: polyrhythmic — polyrhythmic fixture fired "
        f"(conf={result.confidence:.4f}, pair ({e['i']},{e['j']}) = "
        f"{e['p']}:{e['q']}, rates {e['rate_slow']:.3f}/{e['rate_fast']:.3f} Hz, "
        f"ratio dev {e['ratio_dev']*100:.2f}%); "
        f"locked, drifting, two_cluster, phase_beating, flam silent"
    )


# ---------------------------------------------------------------------------
# unstable_bridge — fires only on the dedicated bridge fixture (counterfactual).
#
# This is the project's first counterfactual detector — it routes
# intervention evidence through `sim.ablate.ablate_node`. It is also
# the first detector with an asymmetric input signature (takes the
# baseline config on top of the usual observable state, because the
# ablation path needs the full scene+run spec).
#
# Applicability requires `detect_dominant_cluster` on baseline. Five of
# the six earlier fixtures have no coherent cluster structure at all:
#   - regime_locked: all velocities collapse → sep = 0, no partition.
#   - regime_drifting: no cluster reaches the 0.9 local_r floor.
#   - regime_phase_beating: likewise, no coherent sub-cluster.
#   - regime_flam (N=4): coherent pair exists but sep is below the
#     dominant_cluster gate.
#   - regime_polyrhythmic (N=4): no clean two-way coherent split.
# These skip the ablation loop entirely and stay silent by construction.
#
# regime_two_cluster has `dominant_cluster` firing on baseline (4+4 split
# locked at local_r ≈ 0.99 per cluster) — so the detector DOES enter the
# ablation loop there. It stays silent because ablating any one node
# leaves the other 3 cluster members tightly coupled (observed r_post
# >= 0.98 across every ablation), well above the 0.9 collapse gate. This
# is the right answer: a balanced 4+4 cluster has no single load-bearing
# bridge.
#
# regime_unstable_bridge is the dedicated positive: node 0 (a central
# bridge) holds two geometrically distant, frequency-separated spokes
# into a 3-node lock. Ablating node 0 collapses the trio's local_r from
# ~0.93 to ~0.59. Ablating either spoke leaves the remaining bridge+
# spoke pair locked (r_post > 0.98).

def test_detect_unstable_bridge_across_regimes():
    # (a) fires cleanly on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_unstable_bridge.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_unstable_bridge(cfg, art["theta"], art["phase_vel"], rate)
    assert result.fired, "unstable_bridge should fire on regime_unstable_bridge"
    T = art["theta"].shape[0]
    tail_frames = int(UNSTABLE_BRIDGE_TAIL_S * rate)
    assert result.longest_window_frames == tail_frames
    assert result.windows == [(T - tail_frames, T)]
    # Observed ~0.33 drop (local_r 0.93 → 0.59); assert well above noise.
    assert result.confidence > 0.2, (
        f"expected a material local_r drop (>0.2), got confidence {result.confidence:.4f}"
    )

    # Exactly node 0 contributes — the central bridge. Both spokes and
    # every cluster-B member stay above the 0.9 post-ablation floor.
    evidence = _unstable_bridge_evidence(cfg, art["theta"], art["phase_vel"], rate)
    assert len(evidence) == 1, f"expected one bridge node, got {len(evidence)}: {evidence}"
    e = evidence[0]
    assert e["node"] == 0, f"expected bridge = node 0 (central), got {e['node']}"
    assert e["baseline_local_r"] >= DOMINANT_CLUSTER_MIN_LOCAL_R, (
        f"bridge's baseline cluster must be coherent; got r_pre={e['baseline_local_r']:.4f}"
    )
    assert e["ablated_local_r"] < DOMINANT_CLUSTER_MIN_LOCAL_R, (
        f"bridge ablation must drop local_r below the gate; "
        f"got r_post={e['ablated_local_r']:.4f}"
    )
    assert 0 in set(e["cluster_members"].tolist())

    # (b) silent on locked — dominant_cluster fails precondition
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_unstable_bridge(
            cfg_l, art_l["theta"], art_l["phase_vel"], cfg_l["run"]["control_rate_hz"]
        ),
        "unstable_bridge", "regime_locked",
    )

    # (c) silent on drifting — dominant_cluster fails precondition
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_unstable_bridge(
            cfg_d, art_d["theta"], art_d["phase_vel"], cfg_d["run"]["control_rate_hz"]
        ),
        "unstable_bridge", "regime_drifting",
    )

    # (d) silent on phase_beating — dominant_cluster fails precondition
    cfg_b, art_b = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    _assert_silent(
        detect_unstable_bridge(
            cfg_b, art_b["theta"], art_b["phase_vel"], cfg_b["run"]["control_rate_hz"]
        ),
        "unstable_bridge", "regime_phase_beating",
    )

    # (e) silent on flam — dominant_cluster fails precondition (sep < 5)
    cfg_f, art_f = _run(CONFIGS_DIR / "regime_flam.yaml")
    _assert_silent(
        detect_unstable_bridge(
            cfg_f, art_f["theta"], art_f["phase_vel"], cfg_f["run"]["control_rate_hz"]
        ),
        "unstable_bridge", "regime_flam",
    )

    # (f) silent on polyrhythmic — dominant_cluster fails precondition
    cfg_p, art_p = _run(CONFIGS_DIR / "regime_polyrhythmic.yaml")
    _assert_silent(
        detect_unstable_bridge(
            cfg_p, art_p["theta"], art_p["phase_vel"], cfg_p["run"]["control_rate_hz"]
        ),
        "unstable_bridge", "regime_polyrhythmic",
    )

    # (g) silent on two_cluster — dominant_cluster FIRES on baseline but
    # the 4+4 split has no single load-bearing node. The detector must
    # enter the ablation loop and decide silent honestly.
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    result_t = detect_unstable_bridge(
        cfg_t, art_t["theta"], art_t["phase_vel"], cfg_t["run"]["control_rate_hz"]
    )
    _assert_silent(result_t, "unstable_bridge", "regime_two_cluster")
    ev_t = _unstable_bridge_evidence(
        cfg_t, art_t["theta"], art_t["phase_vel"], cfg_t["run"]["control_rate_hz"]
    )
    assert ev_t == [], (
        f"regime_two_cluster has no bridge node; evidence must be empty, got {ev_t}"
    )

    print(
        f"ok: unstable_bridge — unstable_bridge fixture fired "
        f"(conf={result.confidence:.4f}, bridge=node {e['node']} in cluster "
        f"{e['cluster_idx']}, r_pre={e['baseline_local_r']:.3f} → "
        f"r_post={e['ablated_local_r']:.3f}); locked, drifting, phase_beating, "
        f"flam, polyrhythmic, two_cluster silent"
    )


# ---------------------------------------------------------------------------
# brittle_lock — counterfactual detector: fires when a small whole-run
# frequency nudge to some node collapses a baseline phase lock. Expected
# fire/silent matrix across the current fixtures:
#
#   regime_brittle_lock   : FIRES  (nodes 2, 3, 4 each drop tail-r below 0.9)
#   regime_locked         : silent (robust lock: even 0.25 Hz nudge stays
#                                   ~0.99; lock absorbs the perturbation)
#   regime_drifting       : silent (phase_locked applicability gate fails)
#   regime_phase_beating  : silent (phase_locked applicability gate fails)
#   regime_flam           : silent (phase_locked applicability gate fails)
#   regime_polyrhythmic   : silent (phase_locked applicability gate fails)
#   regime_two_cluster    : silent (phase_locked applicability gate fails)
#   regime_unstable_bridge: silent (phase_locked applicability gate fails)
#
# regime_locked is the interesting honest negative: it passes the
# applicability gate (it IS phase-locked on baseline) but its coupling
# is strong enough that no single-node 0.25 Hz nudge pushes the
# ensemble below the 0.9 gate. Silent here means "the lock is robust",
# not "the detector didn't apply".

def test_detect_brittle_lock_across_regimes():
    # (a) fires cleanly on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_brittle_lock.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_brittle_lock(cfg, art["theta"], art["phase_vel"], rate)
    assert result.fired, "brittle_lock should fire on regime_brittle_lock"
    T = art["theta"].shape[0]
    tail_frames = int(BRITTLE_LOCK_TAIL_S * rate)
    assert result.longest_window_frames == tail_frames
    assert result.windows == [(T - tail_frames, T)]
    # Observed mean drop ≈ 0.11 across nodes 2, 3, 4.
    assert result.confidence > 0.05, (
        f"expected a material tail-r drop (>0.05), got confidence {result.confidence:.4f}"
    )

    # The load-bearing brittle nodes are the three outer-ω nodes (2, 3, 4).
    # Inner-ω nodes (0, 1) stay above the 0.9 gate when nudged — the
    # ensemble recaptures them.
    evidence = _brittle_lock_evidence(cfg, art["theta"], art["phase_vel"], rate)
    brittle_nodes = sorted({int(e["node"]) for e in evidence})
    assert brittle_nodes == [2, 3, 4], (
        f"expected brittle nodes = [2, 3, 4], got {brittle_nodes}"
    )
    for e in evidence:
        assert e["baseline_r"] >= BRITTLE_LOCK_MIN_R, (
            f"baseline r must clear the gate on a phase_locked-applicable run; "
            f"got baseline_r={e['baseline_r']:.4f}"
        )
        assert e["nudged_r"] < BRITTLE_LOCK_MIN_R, (
            f"brittle node {e['node']}: nudged tail-r must fall below the gate; "
            f"got nudged_r={e['nudged_r']:.4f}"
        )
        assert e["delta_hz"] == BRITTLE_LOCK_NUDGE_HZ

    # (b) silent on regime_locked — applicability gate passes but lock
    # is robust; worst-case post-nudge tail-r stays ~0.99.
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    result_l = detect_brittle_lock(
        cfg_l, art_l["theta"], art_l["phase_vel"], cfg_l["run"]["control_rate_hz"]
    )
    _assert_silent(result_l, "brittle_lock", "regime_locked")
    # Evidence must be empty — not just "detector returned silent"; we
    # want the honest-negative path where the loop runs and every node
    # holds the lock.
    ev_l = _brittle_lock_evidence(
        cfg_l, art_l["theta"], art_l["phase_vel"], cfg_l["run"]["control_rate_hz"]
    )
    assert ev_l == [], (
        f"regime_locked is robust; nudge evidence must be empty, got {ev_l}"
    )

    # (c) silent on every phase_locked-silent fixture — applicability gate fails.
    for name in [
        "regime_drifting",
        "regime_phase_beating",
        "regime_flam",
        "regime_polyrhythmic",
        "regime_two_cluster",
        "regime_unstable_bridge",
    ]:
        cfg_x, art_x = _run(CONFIGS_DIR / f"{name}.yaml")
        _assert_silent(
            detect_brittle_lock(
                cfg_x, art_x["theta"], art_x["phase_vel"], cfg_x["run"]["control_rate_hz"]
            ),
            "brittle_lock", name,
        )

    drops = {e["node"]: (e["baseline_r"], e["nudged_r"]) for e in evidence}
    print(
        f"ok: brittle_lock — brittle_lock fixture fired "
        f"(conf={result.confidence:.4f}, nudge=+{BRITTLE_LOCK_NUDGE_HZ:.2f} Hz, "
        f"brittle={brittle_nodes}, "
        f"{'; '.join(f'n{k}: {r0:.3f}→{r1:.3f}' for k, (r0, r1) in drops.items())}); "
        f"locked silent (lock robust); drifting, phase_beating, flam, polyrhythmic, "
        f"two_cluster, unstable_bridge silent (applicability gate)"
    )


if __name__ == "__main__":
    test_detect_phase_locked_across_regimes()
    test_detect_drifting_across_regimes()
    test_detect_phase_beating_across_regimes()
    test_detect_flam_across_regimes()
    test_detect_polyrhythmic_across_regimes()
    test_detect_dominant_cluster_across_regimes()
    test_detect_unstable_bridge_across_regimes()
    test_detect_brittle_lock_across_regimes()
