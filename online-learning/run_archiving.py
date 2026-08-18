"""Per-run archiving for the `mechanism*_repeated_runs.py` harnesses and `mechanism2_statistics.py`.

Every run of those scripts still writes its canonical output to the same flat filename in
`online-learning/` -- `05_online_learning_policy_comparison_exp*.ipynb`, `mechanism2_statistics.py`
and chapter 8 of the thesis all read that fixed path, so it cannot move. What was missing is a
record of *which* run produced the file currently sitting there: every re-run silently overwrote
the previous one, and the only trace of a run's own log was whatever got manually piped to a
`*.log` file (or nothing).

`start_run_archive` gives each run its own timestamped folder under `run_archive/runs/` holding
a full stdout+stderr transcript (`run.log`), and the caller drops a copy of its results there
too. The canonical flat file is untouched; this only adds a second, never-overwritten copy next
to a log of the run that produced it.
"""
import os
import sys
from datetime import datetime


class _Tee:
    """Duplicates every write across `streams` (e.g. the real stdout and a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def start_run_archive(ol_dir, run_name):
    """Create `run_archive/runs/<timestamp>_<run_name>/` and tee stdout/stderr into a `run.log`
    inside it. Call this before any other output is produced. Returns the run directory so the
    caller can also save a copy of its results there."""
    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(ol_dir, "run_archive", "runs", f"{run_ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "run.log"), "w")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return run_dir
