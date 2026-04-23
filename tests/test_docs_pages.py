#!/usr/bin/env python3
"""Artifact-truth smoke for the GitHub Pages bundle under `docs/`.

The committed `docs/` tree is the outward-facing artifact: the live demo
served at https://henry-filgueiras.github.io/resonant-instrument-lab/ is
exactly this directory. These tests lock in the narrow guarantees a cold
reader's first click depends on, without parsing JS or spinning up a
browser:

- Landing page is present and carries the two canonical primary links
  the README and landing copy both feature (one Intervention Atlas
  card, one A/B comparison card).
- Viewer bundle (`docs/demo/`) has its three-file manifest.
- At least one canonical fixture bundle under `docs/runs/demo/<name>/`
  carries the whitelist `scripts/build_pages.py` promises.
- Every local href / img src on the landing page resolves to a file on
  disk, including the query-string parameters that deep-link into the
  viewer (`atlas=`, `summaryA=`, `summaryB=`, `topologyA=`, `topologyB=`).
- `docs/.nojekyll` sentinel is committed so Pages serves the raw tree.

Stdlib-only (`html.parser`, `urllib.parse`, `pathlib`, `json`, `wave`) —
no headless browser, no requests. The link-resolution check is the thin
browser-facing layer: it walks the same hrefs a browser would follow
and asserts the files it would fetch actually exist.

Run directly:
    python tests/test_docs_pages.py
Or via pytest:
    python -m pytest tests/test_docs_pages.py
"""
import json
import wave
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
LANDING = DOCS / "index.html"

# The two hero cards on the landing page. These are the canonical entry
# points the README leads with; if they stop appearing verbatim in the
# rendered HTML, the live demo's first-impression story has drifted.
# href values are what HTMLParser returns (entities already decoded, so
# `&amp;` → `&`).
CANONICAL_ATLAS_HREF = (
    "demo/index.html?atlas=../runs/demo/brittle_lock/atlas.json&select=ablate_n4"
)
CANONICAL_AB_HREF = (
    "demo/index.html"
    "?summaryA=../runs/demo/locked/summary.json"
    "&summaryB=../runs/demo/two_cluster/summary.json"
)

# One fixture whose full browser-asset manifest we smoke-check end to end.
# `brittle_lock` is the smaller of the two atlas hero fixtures (N=5 →
# 10 intervention WAVs) so the test stays cheap; the same manifest shape
# applies to every fixture in FIXTURES.
CANONICAL_FIXTURE = "brittle_lock"

# Mirrors build_pages.FIXTURES. Keep in sync; drift means the landing
# catalogue and the shipped fixture tree have fallen out of step.
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

# Files every per-fixture dir must carry after build_pages.py runs.
# Matches build_pages.FIXTURE_FILES (the browser-consumed whitelist).
FIXTURE_REQUIRED_FILES = (
    "summary.json",
    "topology.json",
    "atlas.json",
    "audio.wav",
    "config.yaml",
)

# Viewer bundle's three-file manifest. Matches build_pages.DEMO_FILES.
DEMO_REQUIRED_FILES = ("index.html", "app.js", "style.css")

# Query-string keys on viewer URLs whose values are relative URLs that
# should resolve to on-disk files. `select=` is an intervention id, not
# a path, so it is deliberately excluded.
URL_VALUED_QUERY_KEYS = frozenset(
    {"atlas", "summaryA", "summaryB", "topologyA", "topologyB"}
)


class _LinkCollector(HTMLParser):
    """Walk an HTML document and collect every local href / img src.

    External URLs (http/https/mailto) are dropped — we only care about
    the links that map to files inside the Pages bundle. For anchors
    the (href, tag, position) is recorded so assertion messages can
    point back at the offending element.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.img_srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a":
            href = attr.get("href")
            if href and _is_local_url(href):
                self.hrefs.append(href)
        elif tag == "img":
            src = attr.get("src")
            if src and _is_local_url(src):
                self.img_srcs.append(src)


def _is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("", "file") and not url.startswith("#")


def _resolve_href(base: Path, href: str) -> Path:
    """Resolve a local href relative to `base` (the HTML file's dir).

    Query and fragment are dropped — the path component is all that
    maps to an on-disk file. Ex: `demo/index.html?atlas=...` → base / "demo/index.html".
    """
    parsed = urlparse(href)
    target = base / parsed.path
    return target.resolve()


def _assert_wav_has_signal(path: Path) -> None:
    """Peak absolute sample > 0 — catches the silent-WAV regression.

    Shares the check used in test_atlas.py so the docs bundle's audio
    artifacts are held to the same audibility bar as the runs/ artifacts
    they were copied from.
    """
    import numpy as np

    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    arr = np.frombuffer(frames, dtype=np.int16)
    peak = int(np.abs(arr).max()) if arr.size else 0
    assert peak > 0, f"{path} is silent (peak sample = 0)"


# ---------------------------------------------------------------------------
# Landing page presence + canonical links.


def test_landing_page_exists_and_has_title():
    """`docs/index.html` must exist and be a real HTML document with the
    documented <title>. Catches the "landing page accidentally removed
    or overwritten with a stub" regression.
    """
    assert LANDING.is_file(), f"landing page missing at {LANDING}"
    html = LANDING.read_text()
    assert "<title>Resonant Instrument Lab — live demo</title>" in html, (
        "landing page <title> has drifted from the canonical copy"
    )
    # landing.css is referenced from the landing page and co-committed.
    assert (DOCS / "landing.css").is_file(), "docs/landing.css missing"
    print(f"ok: landing page — {LANDING.relative_to(REPO)} present with canonical <title>")


def test_landing_page_has_canonical_atlas_link():
    """The landing page must carry the canonical primary atlas link
    (`brittle_lock` preselecting `ablate_n4`). This is one of the two
    hero cards and is what README copy guides the reader toward.
    """
    parser = _LinkCollector()
    parser.feed(LANDING.read_text())
    assert CANONICAL_ATLAS_HREF in parser.hrefs, (
        f"canonical atlas link missing from landing page. "
        f"Expected: {CANONICAL_ATLAS_HREF!r}. "
        f"Got {len(parser.hrefs)} local hrefs."
    )
    print(f"ok: canonical atlas link present — {CANONICAL_ATLAS_HREF}")


def test_landing_page_has_canonical_ab_link():
    """The landing page must carry the canonical A/B comparison link
    (`locked` vs `two_cluster`). This is the secondary hero card and
    the one the dissemination copy references for the dominant_cluster
    flip story.
    """
    parser = _LinkCollector()
    parser.feed(LANDING.read_text())
    assert CANONICAL_AB_HREF in parser.hrefs, (
        f"canonical A/B link missing from landing page. "
        f"Expected: {CANONICAL_AB_HREF!r}. "
        f"Got {len(parser.hrefs)} local hrefs."
    )
    print(f"ok: canonical A/B link present — {CANONICAL_AB_HREF}")


# ---------------------------------------------------------------------------
# Bundle manifest — viewer + at least one fixture.


def test_docs_viewer_bundle_manifest():
    """`docs/demo/` must carry the viewer's three-file manifest and
    nothing else. The pruning half matches build_pages.assemble_bundle's
    whitelist behaviour (stale file regression).
    """
    demo = DOCS / "demo"
    assert demo.is_dir(), f"viewer dir missing at {demo}"
    present = {p.name for p in demo.iterdir()}
    missing = set(DEMO_REQUIRED_FILES) - present
    assert not missing, f"docs/demo missing files: {sorted(missing)}"
    unexpected = present - set(DEMO_REQUIRED_FILES)
    assert not unexpected, (
        f"docs/demo has files outside the whitelist: {sorted(unexpected)} — "
        f"build_pages.py should have pruned them."
    )
    print(f"ok: viewer bundle — docs/demo/ exact manifest: {sorted(present)}")


def test_docs_canonical_fixture_bundle_complete():
    """`docs/runs/demo/<CANONICAL_FIXTURE>/` must carry every file the
    browser reader fetches, and the baseline `audio.wav` + every
    `atlas_audio/<id>.wav` must be a non-silent WAV.

    This is the artifact-truth guarantee: clicking the hero atlas card
    for the canonical fixture cannot land on a broken asset.
    """
    fixture_dir = DOCS / "runs" / "demo" / CANONICAL_FIXTURE
    assert fixture_dir.is_dir(), f"fixture bundle missing at {fixture_dir}"

    missing = [f for f in FIXTURE_REQUIRED_FILES if not (fixture_dir / f).is_file()]
    assert not missing, (
        f"{CANONICAL_FIXTURE}: required files missing from docs bundle: {missing}"
    )

    # atlas.json + atlas_audio/ sanity — the atlas references per-intervention
    # WAVs via audio_path; each referenced WAV must exist and be audible.
    atlas = json.loads((fixture_dir / "atlas.json").read_text())
    atlas_audio_dir = fixture_dir / "atlas_audio"
    assert atlas_audio_dir.is_dir(), (
        f"{CANONICAL_FIXTURE}: atlas_audio/ dir missing under docs/"
    )

    baseline_wav = fixture_dir / atlas["baseline"]["audio_path"]
    assert baseline_wav.is_file(), (
        f"baseline audio_path {atlas['baseline']['audio_path']!r} does not resolve"
    )
    _assert_wav_has_signal(baseline_wav)

    for iv in atlas["interventions"]:
        rel = iv["audio_path"]
        assert rel and rel.startswith("atlas_audio/"), (
            f"{iv['id']}: audio_path shape drifted — got {rel!r}"
        )
        wav = fixture_dir / rel
        assert wav.is_file(), f"{iv['id']}: audio_path points to missing file {wav}"
        _assert_wav_has_signal(wav)

    print(
        f"ok: fixture bundle — {CANONICAL_FIXTURE}: "
        f"{len(FIXTURE_REQUIRED_FILES)} required files + baseline wav + "
        f"{len(atlas['interventions'])} intervention WAVs all present and audible"
    )


def test_docs_nojekyll_sentinel_present():
    """`.nojekyll` must exist so GitHub Pages serves the raw tree without
    running Jekyll over it. `build_pages.py` writes this; regression
    means vanilla filenames still render but any future underscore-
    prefixed file would be hidden.
    """
    sentinel = DOCS / ".nojekyll"
    assert sentinel.is_file(), f".nojekyll sentinel missing at {sentinel}"
    print(f"ok: .nojekyll sentinel present at {sentinel.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Link resolution — the thin browser-facing layer.


def test_landing_page_local_links_resolve_on_disk():
    """Every local href / img src on the landing page must resolve to a
    file that exists under `docs/`, including viewer query-string
    parameters that deep-link into fixture artifacts.

    This is as close as we get to a browser-level check without pulling
    in a headless browser: walk the exact URLs a reader's browser would
    fetch, short of executing the viewer's JS. If this passes, a first-
    click on any landing-page link cannot land on 404.
    """
    parser = _LinkCollector()
    parser.feed(LANDING.read_text())

    # Anchors — resolve the path component; skip external URLs (already
    # filtered by the collector).
    broken: list[tuple[str, str]] = []
    for href in parser.hrefs:
        path_target = _resolve_href(DOCS, href)
        if not path_target.exists():
            broken.append((href, f"path {path_target} missing"))
            continue

        # Query-string values that are themselves relative URLs — resolve
        # them relative to the linked HTML file's directory, the same way
        # the browser's fetch() would.
        parsed = urlparse(href)
        if not parsed.query:
            continue
        query_base = path_target.parent
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key not in URL_VALUED_QUERY_KEYS or not value:
                continue
            resolved = (query_base / value).resolve()
            if not resolved.exists():
                broken.append((href, f"?{key}={value} → {resolved} missing"))

    # Image srcs — straightforward file existence.
    for src in parser.img_srcs:
        target = _resolve_href(DOCS, src)
        if not target.exists():
            broken.append((src, f"img src {target} missing"))

    assert not broken, (
        "landing-page links do not all resolve under docs/:\n  "
        + "\n  ".join(f"{h!r}: {why}" for h, why in broken)
    )
    print(
        f"ok: link resolution — {len(parser.hrefs)} hrefs + "
        f"{len(parser.img_srcs)} img srcs all resolve on disk"
    )


# ---------------------------------------------------------------------------
# Catalogue coverage — every fixture the landing catalogue lists must
# have a real bundle. Narrow assertion: existence of summary.json only;
# the per-file manifest is covered by test_docs_canonical_fixture_bundle_complete
# on the canonical fixture so we don't blow up test cost 8x.


def test_docs_catalogue_fixtures_all_resolve():
    """Every fixture listed in the landing catalogue section must have
    at least a `summary.json` under `docs/runs/demo/<name>/`. The
    canonical-fixture test above covers the full per-file manifest;
    this test covers breadth (every catalogue link leads somewhere).
    """
    missing = []
    for name in FIXTURES:
        summary = DOCS / "runs" / "demo" / name / "summary.json"
        if not summary.is_file():
            missing.append(name)
    assert not missing, (
        f"catalogue fixtures missing summary.json: {missing} — "
        f"run `python scripts/build_pages.py` to regenerate the bundle."
    )
    print(f"ok: catalogue coverage — all {len(FIXTURES)} fixtures have summary.json")


if __name__ == "__main__":
    test_landing_page_exists_and_has_title()
    test_landing_page_has_canonical_atlas_link()
    test_landing_page_has_canonical_ab_link()
    test_docs_viewer_bundle_manifest()
    test_docs_canonical_fixture_bundle_complete()
    test_docs_nojekyll_sentinel_present()
    test_landing_page_local_links_resolve_on_disk()
    test_docs_catalogue_fixtures_all_resolve()
