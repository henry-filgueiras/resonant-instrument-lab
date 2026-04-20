# Ontology — v0 Labels

Implementation-facing reference for the 13 v0 labels. This is the sheet detector code grows against, not a philosophy document. All definitions assume the pulse-driven Coupled Oscillator Garden (see `DESIGN_V0.md §2` and the simulator / detector contracts in `DESIGN_V0.md §9–§10`).

**Kinds.**
- *structural* — about topology / who-is-grouped-with-whom.
- *dynamical* — about how state evolves over time.
- *perceptual* — about how the clip presents to a listener.
- *counterfactual* — requires an in-sim ablation rerun.
- *hybrid* — combinations of the above.

**Confidence is evidence margin**, not epistemic uncertainty (see `DESIGN_V0.md §3.1`). Each entry names its margin source.

## Quick reference

| label              | kind                      | ablation needed | compound |
|--------------------|---------------------------|-----------------|----------|
| `phase_locked`     | dynamical                 | no              | no       |
| `drifting`         | dynamical                 | no              | no       |
| `polyrhythmic`     | structural + dynamical    | no              | no       |
| `phase_beating`    | dynamical + perceptual    | no              | no       |
| `dominant_cluster` | structural + dynamical    | no              | no       |
| `brittle_lock`     | dynamical + counterfactual| yes             | yes (on `phase_locked`) |
| `unstable_bridge`  | structural + counterfactual| yes            | no       |
| `groove`           | dynamical + perceptual    | no              | no       |
| `groove_collapse`  | dynamical (transition)    | no              | yes (on `groove` + `drifting`) |
| `beat_bloom`       | dynamical (transient)     | no              | no       |
| `tension_break`    | dynamical (event-triggered)| no             | no       |
| `flam`             | perceptual                | no              | no       |
| `regime_change`    | structural (transition)   | no              | no       |

"Compound" = depends on other detector outputs as primary evidence. Allowed, but must be declared.

---

## `phase_locked`

- **Kind**: dynamical
- **Musical intuition**: the whole ensemble has fallen into one coherent pulse.
- **Detector sketch**:
  ```python
  r = kuramoto_order(theta_of_t)                         # shape (T,)
  mask = r > cfg.r_lock
  windows = merge_windows(mask, hysteresis_s=cfg.hyst, min_duration_s=cfg.min_dur)
  for w in windows:
      conf = sigmoid(k * (mean(r[w]) - cfg.r_lock)) * min(1.0, duration(w) / cfg.min_dur)
  ```
- **Primary inputs**: `θ(t)`; derived `r(t)` (helper).
- **Knobs**: `r_lock=0.9`, `min_dur=1.0 s`, `hyst=0.2 s`, `k=10`.
- **Confidence source**: window-mean margin of `r` above `r_lock`, scaled by duration.
- **Failure modes**: brief noise dips fragment windows (use hysteresis); a strongly-dominant cluster with outliers can fail global `r` but look locked perceptually — covered by `dominant_cluster`.
- **Overlaps**: `groove` (lower `r` band), `brittle_lock` (counterfactual superset), `dominant_cluster` (partial).

## `drifting`

- **Kind**: dynamical
- **Musical intuition**: no ensemble coherence — everyone doing their own thing.
- **Detector sketch**:
  ```python
  windows = merge_windows(r < cfg.r_drift, min_duration_s=cfg.min_dur)
  conf = sigmoid(k * (cfg.r_drift - mean(r[w]))) * duration_factor
  ```
- **Primary inputs**: `θ(t)` → `r(t)`.
- **Knobs**: `r_drift=0.3`, `min_dur=1.0 s`.
- **Confidence source**: margin of `r` *below* `r_drift`, scaled by duration.
- **Failure modes**: global `r` can be low while sub-clusters are coherent — not a failure, but means `polyrhythmic` may co-fire.
- **Overlaps**: mutually exclusive with `phase_locked`; compatible with `polyrhythmic` (document this as a legitimate co-occurrence).

## `polyrhythmic`

- **Kind**: structural + dynamical
- **Musical intuition**: two or more stable rhythm streams running side by side.
- **Detector sketch**:
  ```python
  labels = cluster_assignments(theta, phase_velocity, t)    # shape (T, N)
  stable = clusters_stable_for(labels, min_fraction=cfg.stability)
  freqs = mean_phase_velocity_per_cluster(labels, t)
  if len(stable) >= 2 and freq_separation_ok(freqs, eps=cfg.freq_eps):
      fire
  ```
- **Primary inputs**: `θ(t)`, `cluster_assignments(t)` (helper).
- **Knobs**: `min_clusters=2`, `stability=0.8`, `freq_eps=0.1` (relative).
- **Confidence source**: cluster stability fraction × min pairwise frequency separation.
- **Failure modes**: DBSCAN sensitivity; near-1:1 frequency ratios trigger false positives; v0 has no rational-ratio metric.
- **Overlaps**: `dominant_cluster` (one polyrhythm stream often dominates); `drifting` (low global `r` can coexist).

## `phase_beating` (was `beating`)

- **Kind**: dynamical + perceptual
- **Musical intuition**: two near-frequency pulse streams whose onsets slowly drift relative to each other — the pulse-domain analogue of acoustic beating.
- **Detector sketch**:
  ```python
  for (i, j) with |f_i - f_j| < cfg.delta_f_max:
      offsets = pulse_arrival_offset_series(pulses_i, pulses_j)   # 1D series
      spec = low_freq_spectrum(offsets)
      if dominant_peak(spec) > cfg.peak_sharpness and n_beats(offsets) >= cfg.min_cycles:
          fire for pair (i, j)
  ```
- **Primary inputs**: `pulse_fired(t)` per node, phase velocities for frequency estimate.
- **Knobs**: `delta_f_max=0.5 Hz`, `min_cycles=2`, `peak_sharpness=<tuned>`.
- **Confidence source**: sharpness / dominance of the low-freq peak in the onset-offset series.
- **Failure modes**: needs enough pulses in the window; tiny Δf → beat period longer than the clip; large Δf → no coherent offset structure.
- **Overlaps**: `flam` (instantaneous snapshot when offset is 10–40 ms); `beat_bloom` (transient form); does **not** claim audio-rate wave interference.

## `dominant_cluster`

- **Kind**: structural + dynamical
- **Musical intuition**: one group is running the show.
- **Detector sketch**:
  ```python
  labels = cluster_assignments(...)
  big = largest_cluster(labels)
  r_local = local_order_parameter(theta, members=big)
  if |big| >= cfg.size_frac * N and mean(r_local[w]) > cfg.r_dom:
      fire, per_node = membership_mask
  ```
- **Primary inputs**: `cluster_assignments(t)`, `θ(t)`.
- **Knobs**: `size_frac=0.5`, `r_dom=0.8`, `min_dur=1.0 s`.
- **Confidence source**: (size margin above `size_frac·N`) × local-`r` margin.
- **Failure modes**: at `N=8` the 50% threshold is coarse (5 vs 4 nodes matters); balanced 4/4 splits never fire — document this edge.
- **Overlaps**: `phase_locked` (when the dominant cluster is the whole ensemble); `polyrhythmic` (one stream dominates two).

## `brittle_lock`

- **Kind**: dynamical + counterfactual
- **Musical intuition**: locked, but a breath would break it.
- **Detector sketch**:
  ```python
  if not phase_locked(run).fired:
      return NOT_FIRED
  for delta in cfg.perturbation_ladder:                 # ascending
      alt = sim.ablate(run.config, {"type": "nudge", "node": ANY, "delta_hz": delta})
      if not phase_locked(alt).fired:
          return FIRED, conf = 1.0 - (delta / max_delta)
  return NOT_FIRED
  ```
- **Primary inputs**: run artifacts + `sim.ablate` rerun API (see `DESIGN_V0.md §9`).
- **Knobs**: `perturbation_ladder=[0.05, 0.1, 0.2, 0.4]` Hz, `max_rollouts=8`, `node_selection_policy` (e.g. highest-contribution node by coupling sum).
- **Confidence source**: inverse of the smallest perturbation that breaks the lock.
- **Failure modes**: expensive; sensitive to ladder choice and which node gets nudged; never fires on non-locked runs by construction.
- **Overlaps**: subset of `phase_locked`; often co-occurs with `tension_break` mid-sweep.

## `unstable_bridge`

- **Kind**: structural + counterfactual
- **Musical intuition**: one node is load-bearing for cluster coherence; remove it and the thing falls apart.
- **Detector sketch**:
  ```python
  dom = dominant_cluster(run)
  if not dom.fired: return NOT_FIRED
  r_pre = mean(local_r(theta, dom.members)[w])
  for i in dom.members:
      alt = sim.ablate(run.config, {"type": "ablate_node", "node": i})
      dom_alt = match_cluster(alt, reference=dom.members)            # documented helper
      r_post = mean(local_r(alt.theta, dom_alt.members)[w])
      if r_pre >= cfg.r_lock > r_post:
          fire for node i, conf = sigmoid(k * (r_pre - r_post))
  ```
- **Primary inputs**: run artifacts + `sim.ablate({"ablate_node"})` rerun API.
- **Knobs**: `r_lock=0.9`, `cluster_match_iou_min=0.5`, `budget=N rollouts`.
- **Confidence source**: magnitude of `local_r` drop under ablation.
- **Failure modes**: cluster identity can flip under ablation — matching is a judgment call (documented helper, not detector-local); O(N) rollouts make it expensive; only meaningful on runs that already have `dominant_cluster`.
- **Overlaps**: `brittle_lock` (different perturbation axis — nudge vs ablate); `dominant_cluster` (its typical context).

## `groove`

- **Kind**: dynamical + perceptual
- **Musical intuition**: a living pocket — loose enough to breathe, tight enough to feel intentional.
- **Detector sketch**:
  ```python
  in_band = (r > cfg.r_lo) & (r < cfg.r_hi)
  var_r = rolling_variance(r, window_s=cfg.var_window)
  breathing = (var_r > cfg.var_lo) & (var_r < cfg.var_hi)
  windows = merge_windows(in_band & breathing, min_duration_s=cfg.min_dur)
  ```
- **Primary inputs**: `θ(t)` → `r(t)`.
- **Knobs**: `r_lo=0.6`, `r_hi=0.85`, `var_window=0.5 s`, `var_lo`, `var_hi` (tuned during sniff test), `min_dur=1.0 s`.
- **Confidence source**: distance of `r` from band edges × fraction of window with `var_r` in the breathing band.
- **Failure modes**: **the weakest detector in v0.** Groove is a perceptual property that a band-pass on `r` only approximates. Expect disagreement with human ears during sniff test; flag disagreements and iterate.
- **Overlaps**: lower neighbor of `phase_locked`; precedes `groove_collapse`.

## `groove_collapse`

- **Kind**: dynamical (transition) — **compound**
- **Musical intuition**: the pocket falls out.
- **Detector sketch**:
  ```python
  for g in groove(run).windows:
      horizon = [g.t_end, g.t_end + cfg.transition_window]
      if any(d overlaps horizon for d in drifting(run).windows):
          fire, window = [g.t_end - eps, g.t_end + cfg.transition_window]
          conf = drop_rate_of_r(t=g.t_end, window=cfg.transition_window)
  ```
- **Primary inputs**: outputs of `groove` and `drifting` detectors (declared compound).
- **Knobs**: `transition_window=1.0 s`, `drop_rate_min`.
- **Confidence source**: suddenness of the `r` drop across the transition.
- **Failure modes**: inherits upstream detector weaknesses (esp. `groove`); transitions through `phase_beating` or brief relocks can miss.
- **Overlaps**: `regime_change` when cluster structure also flips; `tension_break` when triggered by a `setK` event.

## `beat_bloom` (was `interference_bloom`)

- **Kind**: dynamical (transient)
- **Musical intuition**: two streams drift toward coherence, briefly kiss, and drift apart.
- **Detector sketch**:
  ```python
  for (i, j) in candidate_pairs:
      pb = pairwise_local_r(theta_i, theta_j, t)             # scalar(t)
      peaks = find_peaks(pb, prominence=cfg.prom_min)
      for p in peaks:
          if pb[p] > cfg.peak_threshold and return_below(pb, p, cfg.return_thresh, within=cfg.max_bloom_s):
              fire for pair, window = bloom_region(pb, p)
  ```
- **Primary inputs**: `θ(t)`, pairwise local order parameter.
- **Knobs**: `peak_threshold`, `return_thresh`, `max_bloom_s=3.0`, `prom_min`.
- **Confidence source**: peak height × symmetry of the rise/fall.
- **Failure modes**: long-lived pairwise coherence is a lock, not a bloom; very brief excursions are noise; requires the run long enough for a full rise-and-fall.
- **Overlaps**: subset of `phase_beating`; must not co-fire with sustained `phase_locked` on the same pair.

## `tension_break`

- **Kind**: dynamical (event-triggered transition)
- **Musical intuition**: a knob was pushed past a threshold and the rhythm snapped.
- **Detector sketch**:
  ```python
  for ev in events.filter(type == "setK"):
      pre = cluster_signature(run, [ev.t - cfg.pre, ev.t])
      post = cluster_signature(run, [ev.t, ev.t + cfg.post])
      d = 1.0 - signature_similarity(pre, post)
      if d > cfg.dissim_min:
          fire, window = [ev.t - eps, ev.t + cfg.post], conf = d
  ```
- **Primary inputs**: `events.jsonl` (parameter-change events), `cluster_assignments(t)`.
- **Knobs**: `pre=0.5 s`, `post=1.0 s`, `dissim_min=0.4`.
- **Confidence source**: cluster-signature dissimilarity across the event.
- **Failure modes**: needs an explicit parameter-change event in the log — spontaneous bifurcations won't fire; near-events with tiny ΔK may fire spuriously.
- **Overlaps**: strict subset of `regime_change`; often precedes `groove_collapse`.

## `flam`

- **Kind**: perceptual
- **Musical intuition**: two pulses meant to land together but miss by a musically audible sliver.
- **Detector sketch**:
  ```python
  pairs = pair_pulses_across_voices(pulses_by_node, window=cfg.pair_win)
  for (p_a, p_b) in pairs:
      delta = abs(p_a.t - p_b.t)
      if cfg.min_delta < delta < cfg.max_delta:
          tag pair as flammed
  fire per-window, conf = flammed_pair_count / expected_pair_count
  ```
- **Primary inputs**: `pulse_fired(t)` per node, voice assignments.
- **Knobs**: `min_delta=0.010 s`, `max_delta=0.040 s`, `pair_win=0.060 s`.
- **Confidence source**: fraction of nominally-paired pulses falling in the flam window.
- **Failure modes**: "meant to land together" is undefined without a reference — we use `pair_win` proximity as the proxy, which breaks down in dense textures; sub-10 ms offsets register as near-unison, not flam (intentional).
- **Overlaps**: instantaneous form of `phase_beating`.

## `regime_change`

- **Kind**: structural (transition)
- **Musical intuition**: the piece you're hearing has become a different piece.
- **Detector sketch**:
  ```python
  for t in sliding_midpoints(step=cfg.step):
      pre  = cluster_signature(run, [t - cfg.pre, t])
      post = cluster_signature(run, [t, t + cfg.post])
      d = 1.0 - signature_similarity(pre, post)
      if d > cfg.thresh sustained past cfg.hysteresis:
          fire, window = [t - eps, t + cfg.hysteresis]
  # signature = (cluster_count, size_histogram, mean_freq_histogram)
  ```
- **Primary inputs**: `cluster_assignments(t)`, phase velocities.
- **Knobs**: `pre=1.0 s`, `post=1.0 s`, `thresh=0.5`, `hysteresis=0.75 s`, `step=0.25 s`.
- **Confidence source**: dissimilarity magnitude, weighted by post-hysteresis persistence.
- **Failure modes**: slow drift accumulates into false positives; signature choice biases which changes are visible; cluster-finding instability near boundaries.
- **Overlaps**: superset of `tension_break`; often brackets `groove_collapse`.

---

## Known weak spots

Flag but do **not** reopen in this pass; these become sniff-test-driven iterations.

- **`groove`** — band-pass on `r` is a coarse proxy for a perceptual property. Highest-likelihood detector to disagree with human ears during sniff test.
- **`flam`** — "meant to land together" is under-defined; the current proxy (proximity pairing) breaks in dense pulse textures.
- **`polyrhythmic`** — no rational-ratio reasoning in v0 (`freq_eps` alone); 2:3 vs 2:2.95 both fire.
- **`dominant_cluster`** at `N=8` — 50% threshold is coarse; balanced splits never fire, which may mask real structure.
- **Ablation matching (`unstable_bridge`)** — before/after cluster correspondence is a judgment call living in a helper, not in the detector itself; its parameterization is a load-bearing but under-specified choice.
