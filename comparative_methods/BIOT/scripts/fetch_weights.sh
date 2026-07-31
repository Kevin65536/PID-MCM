#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
method_root="$(cd -- "$script_dir/.." && pwd)"
target_dir="$method_root/checkpoints"
revision="d138e32634e52ae9fa6ec98ac9c4087b14ca869a"

mkdir -p "$target_dir"

curl --fail --location --retry 3 --continue-at - \
  --output "$target_dir/EEG-PREST-16-channels.ckpt" \
  "https://raw.githubusercontent.com/ycq091044/BIOT/$revision/pretrained-models/EEG-PREST-16-channels.ckpt"
curl --fail --location --retry 3 --continue-at - \
  --output "$target_dir/EEG-SHHS+PREST-18-channels.ckpt" \
  "https://raw.githubusercontent.com/ycq091044/BIOT/$revision/pretrained-models/EEG-SHHS%2BPREST-18-channels.ckpt"
curl --fail --location --retry 3 --continue-at - \
  --output "$target_dir/EEG-six-datasets-18-channels.ckpt" \
  "https://raw.githubusercontent.com/ycq091044/BIOT/$revision/pretrained-models/EEG-six-datasets-18-channels.ckpt"

(
  cd "$target_dir"
  sha256sum --check <<'CHECKSUMS'
40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70  EEG-PREST-16-channels.ckpt
391b53f77d5060e43746f3ffd5aae8107785f4fd122bf2f87221a1b5b2854a84  EEG-SHHS+PREST-18-channels.ckpt
78ff15a1782f194286a97b1abe68b2bf100a39803325e2917337d0a77a228542  EEG-six-datasets-18-channels.ckpt
CHECKSUMS
)

