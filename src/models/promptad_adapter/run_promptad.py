"""Invoke PromptAD Python scripts from repo root (Stage 1 wrappers)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m src.models.promptad_adapter.run_promptad <script.py> [args...]")
        return 2
    repo_root = Path(__file__).resolve().parents[3]
    external_root = repo_root / "external" / "PromptAD"
    script = external_root / sys.argv[1]
    if not script.exists():
        print(f"missing script: {script}")
        return 1
    cmd = [sys.executable, str(script), *sys.argv[2:]]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{external_root}:{env.get('PYTHONPATH', '')}".strip(":")
    return subprocess.call(cmd, cwd=external_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
