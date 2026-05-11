#!/usr/bin/env bash
# shellcheck shell=bash
# Source from any repo script after REPO_ROOT is set to the absolute repository root.
if [ -z "${REPO_ROOT:-}" ]; then
  echo "ERROR: REPO_ROOT must be set before sourcing scripts/repo_env.sh" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
