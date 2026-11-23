#!/usr/bin/env bash
# Safe update process: test -> paper -> live. Never deploy straight to live.
#   ./scripts/update.sh <git-ref>
# 1. checks out the ref in a fresh worktree, 2. runs the full test suite,
# 3. switches the running bot to PAPER with the new code. Promotion to LIVE is manual and
#    requires the readiness gate to pass again (any strategy/risk/cost change invalidates it).
set -euo pipefail
REF="${1:?usage: update.sh <git-ref>}"
cd "$(dirname "$0")/.."
git fetch --all --tags
git worktree add -f "../sentinel-$REF" "$REF"
( cd "../sentinel-$REF" && python3 -m venv .venv && .venv/bin/pip install -q -e ".[all,dev]" && .venv/bin/python -m pytest -q )
echo "Tests passed for $REF."
echo "Now run it in PAPER for your configured evaluation period:"
echo "  cd ../sentinel-$REF && .venv/bin/sentinel paper"
echo "Promote to LIVE only after: sentinel gate  (all checks PASS)"
