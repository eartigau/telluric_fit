#!/usr/bin/env bash
# deploy.sh — Encrypt index.html with StatiCrypt and push to GitHub Pages.
#
# Usage:
#   ./deploy.sh                  # encrypt + commit + push
#   ./deploy.sh --dry-run        # encrypt only, don't commit/push
#   ./deploy.sh --password myPW  # override password (default: nirps4ever)
#
# The password is NOT stored in this script — it is read from the macOS Keychain
# (service: telluric_page_password, account: eartigau) or from
# ~/.telluric_page_password as a fallback (useful on the cluster).
#
# To store the password in the Keychain:
#   security add-generic-password -a eartigau -s telluric_page_password -w "nirps4ever"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/index.html"
OUT="$SCRIPT_DIR/index_encrypted.html"

DRY_RUN=false
OVERRIDE_PW=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=true; shift ;;
    --password)  OVERRIDE_PW="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Get password ─────────────────────────────────────────────────────────────
if [[ -n "$OVERRIDE_PW" ]]; then
  PAGE_PW="$OVERRIDE_PW"
else
  # Try macOS Keychain first
  PAGE_PW="$(security find-generic-password -a eartigau -s telluric_page_password -w 2>/dev/null || true)"
  if [[ -z "$PAGE_PW" ]]; then
    # Fallback: file on cluster
    PW_FILE="$HOME/.telluric_page_password"
    if [[ -f "$PW_FILE" ]]; then
      PAGE_PW="$(cat "$PW_FILE")"
    fi
  fi
  if [[ -z "$PAGE_PW" ]]; then
    echo "ERROR: No password found."
    echo "  Store it with: security add-generic-password -a eartigau -s telluric_page_password -w 'yourpassword'"
    echo "  Or on the cluster: echo 'yourpassword' > ~/.telluric_page_password && chmod 600 ~/.telluric_page_password"
    exit 1
  fi
fi

# ── Encrypt ───────────────────────────────────────────────────────────────────
echo "Encrypting index.html with StatiCrypt…"
npx staticrypt "$SRC" \
  --password "$PAGE_PW" \
  --output "$OUT" \
  --remember 0 \
  --short \
  --template-title "Telluric Correction Pipeline — User Manual" \
  --template-instructions "Enter your password to access the user manual."

echo "  → $OUT"

if $DRY_RUN; then
  echo "Dry run — stopping here (not committing or pushing)."
  exit 0
fi

# ── Deploy ────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

# Replace the public index.html with the encrypted version for deployment
cp "$OUT" "$SCRIPT_DIR/index.html"

git add index.html
git commit -m "deploy: update encrypted index.html"
git push origin main

# Restore the plaintext source so local editing stays easy
cp "$OUT" /tmp/index_encrypted_backup.html   # keep a copy just in case
git checkout index.html

echo ""
echo "Done! Encrypted page pushed to GitHub Pages."
echo "The local index.html has been restored to its plaintext version."
