#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
method_root="$(cd -- "$script_dir/.." && pwd)"
target_dir="$method_root/checkpoints"
revision="500543c7e30bda1b22bfd51a49301b238dee21fd"
filename="pretrained_weights.pth"

mkdir -p "$target_dir"
curl --fail --location --retry 3 --continue-at - \
  --output "$target_dir/$filename" \
  "https://huggingface.co/weighting666/CBraMod/resolve/$revision/$filename"

(
  cd "$target_dir"
  sha256sum --check <<'CHECKSUMS'
0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178  pretrained_weights.pth
CHECKSUMS
)

