"""Pytest conftest — make sibling test support modules importable.

The batch management test files use flat imports like::

    from _batch_management_test_support import build_progress_result

which works when the results directory is on ``sys.path``. This conftest
ensures that without requiring callers to manipulate ``PYTHONPATH``.
"""

import sys
from pathlib import Path

_results_dir = Path(__file__).resolve().parent
if str(_results_dir) not in sys.path:
    sys.path.insert(0, str(_results_dir))
