"""Quick verification of regenerated cache: check source_eeg for stair-steps."""
import sys, numpy as np
from pathlib import Path

cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "croce_validation/cache/croce_local/highwl_v2/single_trial_mental_arithmetic/subject_1")

npz = cache_dir / f"{cache_dir.name}_cache.npz"
if not npz.exists():
    # Try subjectN_cache.npz
    for f in cache_dir.glob("subject*_cache.npz"):
        npz = f
        break

print(f"Checking: {npz}")
data = np.load(npz, allow_pickle=False)
source_keys = sorted([k for k in data.keys() if k.endswith("/source_eeg")])
total = len(source_keys)
broken = 0

for k in source_keys:
    arr = np.asarray(data[k])
    ch0 = arr[:, 0] if arr.ndim > 1 else arr
    diff = np.abs(np.diff(ch0))
    zeros = np.sum(diff < 1e-12)
    if zeros > len(diff) * 0.5:
        broken += 1
        print(f"  STILL BROKEN: {k}: {zeros}/{len(diff)} zeros")

data.close()
print(f"\nResult: {total - broken}/{total} clean, {broken} broken")
if broken == 0:
    print("PASS: All source_eeg keys are clean (no stair-steps).")
else:
    print("FAIL: Some source_eeg keys still have stair-steps.")
    sys.exit(1)
