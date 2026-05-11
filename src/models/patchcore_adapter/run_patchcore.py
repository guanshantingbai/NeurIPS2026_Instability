"""Thin adapter for invoking PatchCore scripts from section pipelines."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m src.models.patchcore_adapter.run_patchcore <script> [args...]")
        return 2
    repo_root = Path(__file__).resolve().parents[3]
    external_root = repo_root / "external" / "patchcore-inspection"
    script = external_root / sys.argv[1]
    if not script.exists():
        print(f"missing script: {script}")
        return 1
    if script.suffix == ".sh":
        cmd = ["bash", str(script), *sys.argv[2:]]
    else:
        cmd = [sys.executable, str(script), *sys.argv[2:]]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{external_root / 'src'}:{env.get('PYTHONPATH', '')}".strip(":")
    return subprocess.call(cmd, cwd=external_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
