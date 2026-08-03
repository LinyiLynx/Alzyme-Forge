#!/usr/bin/env bash
# Download Boltz-2 cache via hf-mirror (huggingface.co blocked on this server).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${BOLTZ_CACHE:-$ROOT/cache}"
MIRROR="https://hf-mirror.com/boltz-community"

mkdir -p "$CACHE"
cd "$CACHE"

download() {
  local repo="$1" file="$2"
  local url="${MIRROR}/${repo}/resolve/main/${file}"
  if [[ -f "$file" || -f "${file%.tar}" ]]; then
    echo "SKIP existing: $file"
    return 0
  fi
  echo "Downloading $file ..."
  curl -L --retry 5 --retry-delay 5 -C - -o "$file" "$url"
}

download "boltz-2" "mols.tar"
download "boltz-2" "boltz2_conf.ckpt"
download "boltz-2" "boltz2_aff.ckpt"
download "boltz-1" "ccd.pkl"

if [[ ! -d mols ]]; then
  echo "Extracting mols.tar ..."
  tar -xf mols.tar
fi

echo "Boltz cache ready at $CACHE"
ls -lh "$CACHE"
