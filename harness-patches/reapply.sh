#!/usr/bin/env bash
# Re-apply local octop-harness patches into the uv-managed venv.
#
# WHY THIS EXISTS
#   octop-harness is a PyPI dependency, so `uv sync` (or any reinstall) replaces
#   site-packages and silently drops the edits in ../0001-*.patch. Run this after
#   every `uv sync`, `uv pip install`, or venv rebuild.
#
#   The upstream repo is available (https://github.com/TencentCloud/octop-harness),
#   so the durable fix is an upstream PR. These patches are the local stand-in.
#
# USAGE
#   ./reapply.sh          apply patches
#   ./reapply.sh --check  report whether they are already applied (exit 1 if not)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
VENV="$PROJECT_DIR/.venv"
TARGET="$VENV/lib/python3.12/site-packages/octop_harness/backends/workspace.py"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

[[ -f "$TARGET" ]] || {
    echo "[ fail ] harness not found at $TARGET — run 'uv sync' first" >&2
    exit 1
}

applied=0
missing=0
for patch in "$HERE"/[0-9][0-9][0-9][0-9]-*.patch; do
    [[ -e "$patch" ]] || continue
    name="$(basename "$patch")"
    # A patch is applied when its added lines are already present.
    added="$(grep -c '^+' "$patch" 2>/dev/null || echo 0)"
    added=$((added - 1))  # drop the +++ header line
    present=0
    while IFS= read -r line; do
        [[ "$line" == +* ]] || continue
        [[ "$line" == "+++"* ]] && continue
        if grep -qF -- "${line#+}" "$TARGET" 2>/dev/null; then
            present=$((present + 1))
        fi
    done < "$patch"

    if [[ "$present" -ge "$added" && "$added" -gt 0 ]]; then
        echo "[  ok  ] $name already applied ($present/$added markers)"
        applied=$((applied + 1))
    else
        missing=$((missing + 1))
        [[ "$CHECK_ONLY" -eq 1 ]] && {
            echo "[stale] $name NOT applied ($present/$added markers)"
            continue
        }
        if patch -p0 --forward --dry-run "$TARGET" < "$patch" >/dev/null 2>&1; then
            patch -p0 --forward "$TARGET" < "$patch" && echo "[  ok  ] $name applied"
        else
            echo "[ warn ] $name did not apply cleanly — review by hand" >&2
            echo "         $TARGET" >&2
        fi
    fi
done

if [[ "$CHECK_ONLY" -eq 1 && "$missing" -gt 0 ]]; then
    echo "run ./reapply.sh to restore $missing patch(es)"
    exit 1
fi
echo "harness patches: $applied already applied, $missing applied now"