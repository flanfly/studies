"""test_config_fields_used.py — every Config field must be read somewhere.

A declared-but-unread config field is worse than a hardcoded constant: it makes
a sweep look like it ran. For each field of the Config dataclass, search the
codebase (repo *.py excluding config.py itself and .venv) for a read — either
`cfg.<field>` or `.<field>` in a broader sense — and fail listing any field with
zero readers.

Phase 0.1 remediation: tskd_diff_order and cpvv_window were orphans; they are
now wired in factors.py (TSKD weekly difference, CPVv window). book_terminal_return
is read in resample.py/preprocess.py.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config  # noqa: E402

# fields intentionally consumed via class methods or metadata rather than
# `cfg.<field>` outside config.py: factors is read by Config.factor_list();
# nprocs is compute metadata read by build_daily/preprocess.
_METADATA = {"factors", "nprocs"}


def code_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".venv", "cache", "__pycache__", "out")]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return out


def test_every_config_field_is_read():
    cfg_fields = [f.name for f in fields(Config)]
    files = code_files()
    src = "\n".join(open(f, encoding="utf-8").read() for f in files
                    if not f.endswith(os.path.join("config.py")))
    orphans = []
    for name in cfg_fields:
        if name in _METADATA:
            continue
        # a read is cfg.<name> anywhere outside config.py
        pat = re.compile(rf"cfg\.{name}\b")
        if not pat.search(src):
            orphans.append(name)
    assert not orphans, f"Config fields declared but never read: {orphans}"


def test_orphan_fields_are_wired():
    """Regression: tskd_diff_order / cpvv_window / book_terminal_return must be
    read in factors.py / resample.py."""
    factors_src = open(os.path.join(ROOT, "factors.py"), encoding="utf-8").read()
    resample_src = open(os.path.join(ROOT, "resample.py"), encoding="utf-8").read()
    assert "cfg.tskd_diff_order" in factors_src
    assert "cfg.cpvv_window" in factors_src
    assert "book_terminal_return" in resample_src


if __name__ == "__main__":
    test_every_config_field_is_read()
    test_orphan_fields_are_wired()
    print("config field audit: PASSED")