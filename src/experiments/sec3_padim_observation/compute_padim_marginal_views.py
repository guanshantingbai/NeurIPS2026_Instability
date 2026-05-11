"""Section 3.1.2 marginal views — not used by default fast path (see run.sh).

Bundled stubs live under samples/fastpath/. Full PaDiM marginal analysis requires FULL_RUN=1.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "compute_padim_marginal_views.py is not part of the fast path. "
        "Use src/experiments/sec3_padim_observation/run.sh (default) or FULL_RUN=1 with PADIM_* and scripts/run_padim_raw.sh (see docs/FULLPATH_PADIM.md)."
    )


if __name__ == "__main__":
    main()
