#!/usr/bin/env python3
"""Validate a v0 simulator run config against DESIGN_V0.md §9.2.

Usage: python scripts/validate_config.py [path/to/config.yaml]
If no path is given, defaults to configs/regime_drifting.yaml.
Exits 0 on ok; nonzero with a precise diagnostic on failure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sim.config import SCHEMA_VERSION, ConfigError, load  # noqa: E402


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/regime_drifting.yaml")
    try:
        load(path)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    except ConfigError as e:
        sys.exit(f"error: {e}")
    except yaml.YAMLError as e:
        sys.exit(f"error: YAML parse failure in {path}: {e}")
    print(f"ok: {path} validates against schema v{SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
