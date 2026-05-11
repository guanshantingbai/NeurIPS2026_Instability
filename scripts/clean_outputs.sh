#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

rm -rf outputs/figures/* outputs/tables/* outputs/cached_results/*
mkdir -p outputs/figures outputs/tables outputs/cached_results
for d in outputs/figures outputs/tables outputs/cached_results; do
  touch "$d/.gitkeep"
done
echo "cleaned generated outputs (restored .gitkeep markers)"
