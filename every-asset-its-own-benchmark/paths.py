"""paths.py — repo-root-relative paths for cache, registry, scratch, outputs.

The repo previously lived at /home/kai/studies/bbb and was renamed. Keep every
path rooted on the real repo directory so renames/moves do not silently break
the pipeline. Heavy data files (5m parquet, scratch) stay on the node volume.
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# pipeline caches under the repo
CACHE_ROOT = os.path.join(REPO_ROOT, "cache")

# heavy per-symbol scratch on the node volume
SCRATCH = "/home/kai/node/data/studies/bbb_scratch"

# experiment outputs & registry under the repo
OUT = os.path.join(REPO_ROOT, "experiments", "out")
REGISTRY = os.path.join(REPO_ROOT, "registry.csv")