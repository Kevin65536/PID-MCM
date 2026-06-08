"""Comprehensive verification of all regenerated caches.
Checks: stair-step artifacts, NaN/Inf, basic statistics, field presence.
"""
import json, sys, numpy as np
from pathlib import Path

CACHE_ROOT = Path("croce_validation/cache/croce_local/highwl_v2")
CANONICAL_FIELDS = [
    "source_eeg", "obs_eeg",
    "source_fnirs_optical_channel_0", "obs_fnirs_optical_channel_0",
    "source_fnirs_optical_channel_1", "obs_fnirs_optical_channel_1",
]


def check_subject_cache(npz_path: Path) -> dict:
    """Check a single subject-level cache npz for issues."""
    result = {"path": str(npz_path), "n_keys": 0, "source_eeg_broken": 0,
              "source_eeg_total": 0, "nan_keys": [], "inf_keys": [],
              "missing_fields": set(), "empty_arrays": []}
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        result["load_error"] = str(e)
        return result

    keys = sorted(data.keys())
    result["n_keys"] = len(keys)

    # Check for missing canonical fields per anchor
    anchors = set()
    for k in keys:
        parts = k.split("/")
        if len(parts) >= 2:
            anchors.add(parts[0])
    for anchor in anchors:
        for field in CANONICAL_FIELDS:
            # Check if any key matches anchor/*/field or anchor/field
            matching = [k for k in keys if k.startswith(anchor + "/") and k.endswith("/" + field)]
            if not matching:
                result["missing_fields"].add(f"{anchor}/{field}")

    for k in keys:
        try:
            arr = np.asarray(data[k])
        except Exception as e:
            result["load_error"] = f"{k}: {e}"
            continue

        if arr.size == 0:
            result["empty_arrays"].append(k)
            continue

        if np.any(np.isnan(arr)):
            result["nan_keys"].append(k)
        if np.any(np.isinf(arr)):
            result["inf_keys"].append(k)

        # Stair-step check for source_eeg
        if k.endswith("/source_eeg"):
            result["source_eeg_total"] += 1
            ch0 = arr[:, 0] if arr.ndim > 1 else arr
            if len(ch0) > 1:
                diff = np.abs(np.diff(ch0))
                zeros = int(np.sum(diff < 1e-12))
                if zeros > len(diff) * 0.5:
                    result["source_eeg_broken"] += 1

    data.close()
    return result


def check_event_cache(npz_path: Path) -> dict:
    """Check a single event-level cache npz."""
    r = check_subject_cache(npz_path)
    r["type"] = "event"
    return r


def walk_caches(root: Path):
    """Walk cache directory and yield all cache npz files."""
    for npz in sorted(root.rglob("*.npz")):
        if "cache.npz" in npz.name or "_cache.npz" in npz.name:
            yield npz


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = CACHE_ROOT

    print(f"Scanning: {root}")
    caches = list(walk_caches(root))
    print(f"Found {len(caches)} cache files\n")

    subject_results = []
    event_results = []
    total_broken = 0
    total_source_eeg = 0
    nan_total = 0
    inf_total = 0

    for npz_path in caches:
        rel = npz_path.relative_to(root)
        is_event = npz_path.parent.name.startswith("event_")
        result = check_subject_cache(npz_path) if not is_event else check_event_cache(npz_path)

        if is_event:
            event_results.append(result)
        else:
            subject_results.append(result)

        total_broken += result["source_eeg_broken"]
        total_source_eeg += result["source_eeg_total"]

        issues = []
        if result.get("load_error"):
            issues.append(f"LOAD ERROR: {result['load_error']}")
        if result["source_eeg_broken"] > 0:
            issues.append(f"STILL BROKEN: {result['source_eeg_broken']}/{result['source_eeg_total']} source_eeg")
        if result["nan_keys"]:
            nan_total += len(result["nan_keys"])
            issues.append(f"NaN in {len(result['nan_keys'])} keys")
        if result["inf_keys"]:
            inf_total += len(result["inf_keys"])
            issues.append(f"Inf in {len(result['inf_keys'])} keys")

        if issues:
            print(f"  [{rel}] {' | '.join(issues)}")

    print(f"\n{'='*60}")
    print(f"SUBJECT-LEVEL: {len(subject_results)} caches")
    print(f"EVENT-LEVEL:   {len(event_results)} caches")
    print(f"TOTAL:         {len(caches)} caches")
    print(f"\nsource_eeg check: {total_broken}/{total_source_eeg} broken ({100*total_broken/max(1,total_source_eeg):.1f}%)")
    print(f"NaN keys: {nan_total}")
    print(f"Inf keys: {inf_total}")

    if total_broken == 0 and nan_total == 0 and inf_total == 0:
        print("\n✅ ALL CACHES PASS")
    else:
        print(f"\n❌ ISSUES FOUND: {total_broken} broken source_eeg, {nan_total} NaN, {inf_total} Inf")


if __name__ == "__main__":
    main()
