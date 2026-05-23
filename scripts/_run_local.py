"""Launcher that runs the LOCAL icu_benchmarks (this worktree) instead of the
pip editable install (which is hard-pinned to the `paper` worktree via a
MetaPathFinder hook).

Usage:
    python _run_local.py <args forwarded to icu_benchmarks.run:main>
"""

import sys
from pathlib import Path

CH_ROOT = Path(__file__).resolve().parent.parent  # /.../YAIB/ch_input_num

sys.meta_path = [
    f for f in sys.meta_path
    if getattr(type(f), "__module__", "") != "__editable___yaib_0_3_1_finder"
]
sys.path.insert(0, str(CH_ROOT))

from icu_benchmarks.run import main  # noqa: E402

if __name__ == "__main__":
    main()
