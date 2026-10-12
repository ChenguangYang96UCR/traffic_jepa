#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="$PROJECT_DIR/upstream/TS_JEPA"
REPOSITORY="https://github.com/Sennadir/TS_JEPA.git"
COMMIT="830ee5c3a7d4e85623bd5a07992732be7fda9bc2"

if [[ -f "$TARGET/src/models/encoder.py" ]]; then
  echo "TS-JEPA already available: $TARGET"
  exit 0
fi

if [[ -e "$TARGET" ]]; then
  echo "ERROR: $TARGET exists but is not a complete TS-JEPA checkout." >&2
  echo "Move it aside, then rerun this script." >&2
  exit 2
fi

mkdir -p "$(dirname "$TARGET")"
git clone --no-checkout "$REPOSITORY" "$TARGET"
git -C "$TARGET" checkout --detach "$COMMIT"

echo "Installed official TS-JEPA commit $COMMIT at $TARGET"
