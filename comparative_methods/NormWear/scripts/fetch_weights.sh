#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
method_root="$(cd -- "$script_dir/.." && pwd)"
target_dir="$method_root/checkpoints"
release_root="https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/download/v1.0.0-alpha"

mkdir -p "$target_dir"
curl --fail --location --retry 3 --continue-at - \
  --output "$target_dir/normwear_pretrain_ckpt.pth" \
  "$release_root/normwear_pretrain_ckpt.pth"
curl --fail --location --retry 3 --continue-at - \
  --output "$target_dir/normwear_msitf_zeroshot_last_checkpoint-5.pth" \
  "$release_root/normwear_msitf_zeroshot_last_checkpoint-5.pth"

(
  cd "$target_dir"
  sha256sum --check <<'CHECKSUMS'
36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561  normwear_pretrain_ckpt.pth
6605c137d3cea7cab479d8ab0f9bc06cc76f1ecd7429f33eb81a6f35e0887dbe  normwear_msitf_zeroshot_last_checkpoint-5.pth
CHECKSUMS
)

