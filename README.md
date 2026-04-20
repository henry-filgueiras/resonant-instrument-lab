# Resonant Instrument Lab

A local research prototype for teaching small language models to **observe, describe, diagnose, and steer** emergent musical systems — starting from the raw continuous signals they produce.

## Why

Most music AI is evaluated by taste. Taste is slow, subjective, and unfalsifiable for a single researcher. This project swaps the target: we work in a **bounded synthetic world** — a Coupled Oscillator Garden — whose dynamics are known and whose semantic labels (*phase-locked*, *groove*, *brittle lock*, *tension break*, and so on) are computable directly from state.

The headline evaluation principle is **intervention grounding**:

> When the model proposes a change, we *apply that change* in the simulator. Did the world move the way the model said it would?

That turns *"does the model understand?"* into a test a laptop can run.

## What v0 is

- A coupled-oscillator simulator with pulse-driven audio and detector-grounded labels.
- A small instruction-tuned model — encoders + frozen base LM + LoRA adapters — that learns to caption, compare, diagnose, and suggest single interventions.
- Fully local, reproducible from seeds, no third-party audio data.

## Current milestone

**Simulator + ground-truth detectors + dataset dump. No model code yet.**

Detectors are the hardest part of the project and must pass a human sniff test before any training is meaningful. Acceptance criteria for this milestone live in [`DESIGN_V0.md §9`](./DESIGN_V0.md#9-first-coding-milestone).

## Docs

- [`DESIGN_V0.md`](./DESIGN_V0.md) — full v0 design note.
- [`DIRECTORS_NOTES.md`](./DIRECTORS_NOTES.md) — project canon and append-only pivot log.

## Status

v0, pre-implementation. Design-only repo.
