#!/usr/bin/env bash
# Creates this repo's own venv, deliberately separate from whatever venv
# micard-design-team (or any other caller) uses -- see README.md's
# "Install" section for why: torch is a heavy, version-pinned dependency,
# and isolating it here keeps a caller's own venv light for anyone who
# never touches this tool. design_team's design_optimization Tool invokes
# this venv's python via subprocess (see its own docstring), never imports
# this package directly into its own process.
#
# Installs the "ppo" extra (torch) too, so the ppo method works out of the
# box in this dedicated venv -- the whole point of isolating it here rather
# than installing it into a caller's own venv.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[ppo]"

echo "== done: $(pwd)/.venv =="
