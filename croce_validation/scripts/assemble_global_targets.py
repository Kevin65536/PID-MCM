"""Assemble per-anchor cache entries into global full-channel targets.

Each anchor's particle filter solves a local state-space model around one
fNIRS optode position.  The anchor-scoped cache stores:

* fNIRS: the anchor's own channel only (1 spatial position, both optical
  channels).  The 36 anchors collectively cover all 36 spatial positions.
* EEG: a local neighbourhood (≈6 channels).  Overlaps between neighbouring
  anchors are resolved by a nearest-anchor rule.

This script reads a per-subject cache and produces a global target file
with the same channel dimensionality as the original recordings:
  EEG  (T, 30)        — one column per 10-20 electrode
  fNIRS (T, 36) × 2   — one column per optode pair, highWL and lowWL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_adjacency import (
    build_channel_adjacency,
    canonicalize_channel_label,
    strip_fnirs_chromophore_suffix,
)
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble per-anchor cache into global targets.")
    p.add_argument("--cache-path", required=True, type=Path, help="Path to subjectX_cache.npz")
    p.add_argument("--manifest-path", type=Path, help="Path to cache_manifest.json (inferred if omitted)")
    p.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    p.add_argument("--output-path", type=Path, help="Output .npz path (default: cache_dir/global_targets.npz)")
    p.add_argument("--use-artifact-eeg", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anchor_base_from_primary(channel_name: str) -> str:
    """Extract the spatial base name from a primary fNIRS channel name."""
    return strip_fnirs_chromophore_suffix(channel_name)


def _build_fnirs_spatial_index(
    paired_bases: List[str],
) -> Dict[str, int]:
    """Map canonicalised fNIRS base name → column index (0..35)."""
    return {canonicalize_channel_label(b): i for i, b in enumerate(paired_bases)}


def _build_eeg_anchor_map(
    anchors: List[str],
    local_eeg_names: Dict[str, List[str]],
    eeg_channel_names: List[str],
    adjacency: Any,
) -> Dict[str, Tuple[str, int]]:
    """For each global EEG channel, pick the nearest covering anchor.

    Returns
    -------
    mapping : dict
        ``{eeg_channel_name: (anchor_key, column_index_in_anchor)}``
    """
    # Anchors are identified by their cache key prefix (e.g. "AF7Fp1")
    anchor_eeg_set: Dict[str, set] = {}
    for anchor in anchors:
        anchor_eeg_set[anchor] = set(local_eeg_names.get(anchor, []))

    # Build anchor → position lookup
    anchor_positions: Dict[str, np.ndarray] = {}
    for idx, base_name in enumerate(adjacency._fnirs_paired_bases if hasattr(adjacency, '_fnirs_paired_bases') else []):
        # We don't have direct access to _fnirs_paired_bases, use the adjacency's fnirs_channel_names
        pass

    # Use fnirs_channel_positions as proxy for anchor positions
    # Each anchor = an fNIRS spatial position; find its index in paired bases
    fnirs_names = list(adjacency.fnirs_channel_names)
    fnirs_positions = np.asarray(adjacency.fnirs_channel_positions_2d, dtype=np.float64)

    # For each anchor, get the position of its primary channel
    anchor_pos: Dict[str, np.ndarray] = {}
    for anchor in anchors:
        for i, name in enumerate(fnirs_names):
            base = canonicalize_channel_label(strip_fnirs_chromophore_suffix(name))
            if base == canonicalize_channel_label(anchor):
                anchor_pos[anchor] = fnirs_positions[i]
                break

    # For each EEG channel, find covering anchors and pick nearest
    eeg_positions = np.asarray(adjacency.eeg_positions_2d, dtype=np.float64)
    mapping: Dict[str, Tuple[str, int]] = {}

    for eeg_idx, eeg_name in enumerate(eeg_channel_names):
        eeg_pos = eeg_positions[eeg_idx]
        best_anchor = None
        best_dist = float("inf")
        best_col = -1

        for anchor, covered_names in anchor_eeg_set.items():
            if eeg_name not in covered_names:
                continue
            # Find column index of this EEG channel within the anchor's local set
            col = list(covered_names).index(eeg_name) if eeg_name in covered_names else -1
            # But covered_names is a set — sets are unordered.  We need the ordered list.
            # Fix below.

        # Second pass with ordered lists
        for anchor in anchors:
            ordered_names = local_eeg_names.get(anchor, [])
            if eeg_name not in ordered_names:
                continue
            if anchor not in anchor_pos:
                continue
            dist = float(np.linalg.norm(anchor_pos[anchor] - eeg_pos))
            if dist < best_dist:
                best_dist = dist
                best_anchor = anchor
                best_col = ordered_names.index(eeg_name)

        if best_anchor is None:
            # Fallback: pick the geometry-closest anchor regardless of coverage
            for anchor in anchors:
                if anchor not in anchor_pos:
                    continue
                ordered_names = local_eeg_names.get(anchor, [])
                if not ordered_names:
                    continue
                dist = float(np.linalg.norm(anchor_pos[anchor] - eeg_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_anchor = anchor
                    best_col = 0  # use first EEG channel as fallback

        if best_anchor is not None:
            mapping[eeg_name] = (best_anchor, best_col)

    return mapping


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble_subject(
    cache_path: Path,
    manifest: Dict[str, Any],
    adjacency: Any,
) -> Dict[str, np.ndarray]:
    """Assemble per-anchor cache into global full-channel targets.

    Returns a dict of {field_name: array} where arrays have shape
    (T, n_full_channels).
    """
    data = dict(np.load(cache_path, allow_pickle=True))
    all_keys = sorted(data.keys())

    # Parse anchor keys and event structure
    anchor_events: Dict[str, List[str]] = {}  # anchor → [event_part, ...]
    for key in all_keys:
        parts = key.split("/")
        if len(parts) == 3:
            anchor, event_part, field = parts
            anchor_events.setdefault(anchor, [])
            if event_part not in anchor_events[anchor]:
                anchor_events[anchor].append(event_part)
        elif len(parts) == 2:
            anchor, field = parts
            anchor_events.setdefault(anchor, ["__no_event__"])

    anchors = sorted(anchor_events.keys())

    # ---- fNIRS spatial index ----
    # Build paired_bases from the manifest's anchor_names
    anchor_names_from_manifest = manifest.get("anchor_names", anchors)
    paired_bases = sorted(set(
        canonicalize_channel_label(a) for a in anchor_names_from_manifest
    ))
    # Preserve the manifest ordering if available
    if "anchor_names" in manifest:
        paired_bases = []
        for a in manifest["anchor_names"]:
            c = canonicalize_channel_label(a)
            if c not in paired_bases:
                paired_bases.append(c)
    fnirs_spatial = {b: i for i, b in enumerate(paired_bases)}

    # ---- Collect per-anchor local EEG names from manifest assembly_meta ----
    assembly_meta = manifest.get("assembly_meta", {})
    local_eeg_names: Dict[str, List[str]] = {}
    for key, meta in assembly_meta.items():
        anchor = key.split("/")[0]
        if "local_eeg_channel_names" in meta:
            local_eeg_names[anchor] = list(meta["local_eeg_channel_names"])

    # ---- Build EEG assembly map ----
    eeg_global_names = list(adjacency.eeg_channel_names)
    eeg_anchor_map = _build_eeg_anchor_map(
        anchors, local_eeg_names, eeg_global_names, adjacency,
    )

    # ---- fNIRS spatial layout ----
    n_fnirs_positions = len(paired_bases)

    # ---- Assemble per event ----
    assembled: Dict[str, Dict[str, np.ndarray]] = {}  # event_part → {field: array}

    for anchor in anchors:
        anchor_base = canonicalize_channel_label(anchor)
        fnirs_idx = fnirs_spatial.get(anchor_base)
        if fnirs_idx is None:
            print(f"  Warning: anchor {anchor} not found in fNIRS spatial index, skipping")
            continue

        for event_part in anchor_events[anchor]:
            prefix = f"{anchor}/{event_part}/"
            if event_part not in assembled:
                assembled[event_part] = {}

            # Read anchor-scoped targets from dict
            _read = lambda f: data.get(prefix + f)

            source_eeg = _read("source_eeg")
            obs_eeg = _read("obs_eeg")
            source_p0 = _read("source_fnirs_optical_channel_0")
            obs_p0 = _read("obs_fnirs_optical_channel_0")
            source_p1 = _read("source_fnirs_optical_channel_1")
            obs_p1 = _read("obs_fnirs_optical_channel_1")

            if source_eeg is None:
                continue
            source_eeg = np.asarray(source_eeg, dtype=np.float64)
            obs_eeg = np.asarray(obs_eeg, dtype=np.float64)
            source_p0 = np.asarray(source_p0, dtype=np.float64)
            obs_p0 = np.asarray(obs_p0, dtype=np.float64)
            source_p1 = np.asarray(source_p1, dtype=np.float64)
            obs_p1 = np.asarray(obs_p1, dtype=np.float64)

            n_eeg = source_eeg.shape[0]
            n_fnirs = source_p0.shape[0]

            # Initialise global arrays on first anchor
            for field, shape, dtype in [
                ("source_eeg", (n_eeg, len(eeg_global_names)), np.float64),
                ("obs_eeg", (n_eeg, len(eeg_global_names)), np.float64),
                ("source_fnirs_optical_channel_0", (n_fnirs, n_fnirs_positions), np.float64),
                ("obs_fnirs_optical_channel_0", (n_fnirs, n_fnirs_positions), np.float64),
                ("source_fnirs_optical_channel_1", (n_fnirs, n_fnirs_positions), np.float64),
                ("obs_fnirs_optical_channel_1", (n_fnirs, n_fnirs_positions), np.float64),
            ]:
                if field not in assembled[event_part]:
                    assembled[event_part][field] = np.full(shape, np.nan, dtype=dtype)

            # ---- Place fNIRS targets ----
            assembled[event_part]["source_fnirs_optical_channel_0"][:, fnirs_idx] = source_p0[:, 0]
            assembled[event_part]["obs_fnirs_optical_channel_0"][:, fnirs_idx] = obs_p0[:, 0]
            assembled[event_part]["source_fnirs_optical_channel_1"][:, fnirs_idx] = source_p1[:, 0]
            assembled[event_part]["obs_fnirs_optical_channel_1"][:, fnirs_idx] = obs_p1[:, 0]

            # ---- Place EEG targets ----
            local_names = local_eeg_names.get(anchor, [])
            for local_col, ch_name in enumerate(local_names):
                if ch_name in eeg_anchor_map:
                    best_anchor, best_col = eeg_anchor_map[ch_name]
                    if best_anchor == anchor and best_col == local_col:
                        assembled[event_part]["source_eeg"][:, eeg_global_names.index(ch_name)] = source_eeg[:, local_col]
                        assembled[event_part]["obs_eeg"][:, eeg_global_names.index(ch_name)] = obs_eeg[:, local_col]

    return assembled


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    # Resolve manifest
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        manifest_path = cache_path.parent / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Resolve subject id and build adjacency
    subject_id = manifest.get("config", {}).get("subject_id", 1)
    subject_id = int(subject_id) if subject_id is not None else 1

    # Minimal dataset to get channel names
    ds = MultiModalEEGfNIRSDataset(
        data_root=args.data_root,
        subject_ids=[subject_id],
        task="motor_imagery",
        window_duration_s=2.5,
        normalize=False,
        normalization_mode="none",
        use_artifact_data=bool(args.use_artifact_eeg),
        exclude_eog=True,
        hbo_only=False,
        hbr_only=False,
    )
    adjacency = build_channel_adjacency(
        "eeg_fnirs_single_trial",
        args.data_root,
        ds.get_eeg_channel_names(),
        ds.get_fnirs_channel_names(),
        reference_subject_id=subject_id,
        use_artifact_data=bool(args.use_artifact_eeg),
    )

    print(f"Assembling global targets for subject {subject_id} ...")
    assembled = assemble_subject(cache_path, manifest, adjacency)

    # Save
    output_path = args.output_path or (cache_path.parent / "global_targets.npz")
    save_dict: Dict[str, np.ndarray] = {}
    for event_part, fields in assembled.items():
        for field_name, array in fields.items():
            key = f"{event_part}/{field_name}" if event_part != "__no_event__" else field_name
            save_dict[key] = array.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    n_events = len(assembled)
    print(f"  {n_events} event(s) assembled")
    print(f"  EEG:    {save_dict.get(list(save_dict.keys())[0], np.array([])).shape}")
    for k, v in save_dict.items():
        if "fnirs" in k and "channel_0" in k:
            print(f"  fNIRS:  {v.shape}")
            break
    print(f"  Saved:  {output_path} ({size_mb:.1f} MB)")

    # Save assembly metadata
    meta_path = output_path.with_suffix(".json")
    assembly_meta = {
        "source_cache": str(cache_path),
        "subject_id": subject_id,
        "n_events": n_events,
        "eeg_channels": list(adjacency.eeg_channel_names),
        "fnirs_spatial_positions": list(manifest.get("anchor_names", [])),
        "eeg_assembly_rule": "nearest_anchor",
        "fnirs_assembly_rule": "direct_anchor_mapping",
    }
    meta_path.write_text(json.dumps(assembly_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Meta:   {meta_path}")


if __name__ == "__main__":
    main()
