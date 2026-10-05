#!/usr/bin/env bash
# Pull DVC-tracked artifacts from the default remote.
#
# A missing remote config, a missing or empty local remote directory, or a repo with no
# DVC-tracked outputs is a no-op (exit 0) with a message, so a fresh clone works offline.
# A non-local remote (e.g. S3, set in .dvc/config.local by Terraform) is pulled normally
# and its errors are surfaced.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0"/..)"
DVC=(uv run --quiet dvc)

skip() { echo "data-pull: $1; nothing to pull (no-op)."; exit 0; }

remote="$("${DVC[@]}" remote default 2>/dev/null || true)"
[[ -n "$remote" ]] || skip "no default DVC remote configured"

url="$("${DVC[@]}" config "remote.$remote.url" 2>/dev/null || true)"
[[ -n "$url" ]] || skip "default remote '$remote' has no URL"
# Relative local paths in .dvc/config and .dvc/config.local are relative to .dvc/.
if [[ "$url" != *://* && "$url" != /* ]]; then
  url="$(realpath -m ".dvc/$url")"
fi

if [[ "$url" != *://* ]]; then
  [[ -d "$url" ]] || skip "local remote '$remote' ($url) does not exist"
  [[ -n "$(ls -A "$url")" ]] || skip "local remote '$remote' ($url) is empty"
fi

if [[ -z "$(git ls-files -- '*.dvc' 'dvc.lock')" ]]; then
  skip "no DVC-tracked outputs in this repo yet"
fi

echo "data-pull: pulling from remote '$remote' ($url)"
"${DVC[@]}" pull
