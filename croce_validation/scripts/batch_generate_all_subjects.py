"""Batch generate source/observation target cache for all 29 subjects.

Calls generate_target_cache.py once per subject with the same configuration.
Uses sequential subject processing (each subject uses internal parallelism for
its 36 anchors).

Usage:
    python croce_validation/scripts/batch_generate_all_subjects.py \
        --parallel-workers 18 --threads 2 --solver-kernel rk4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_SCRIPT = PROJECT_ROOT / "croce_validation" / "scripts" / "generate_target_cache.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch generate target cache for all 29 subjects."
    )
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "EEG+NIRS Single-Trial"))
    p.add_argument("--segment-start-s", type=float, default=60.0)
    p.add_argument("--segment-duration-s", type=float, default=120.0)
    p.add_argument("--use-artifact-eeg", action="store_true", default=True)
    p.add_argument("--num-particles", type=int, default=224)
    p.add_argument("--sigma-prop", type=float, default=5.0)
    p.add_argument("--sigma-nirs", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--parallel-workers", type=int, default=18)
    p.add_argument("--solver-kernel", default="rk4",
                    choices=["expm", "euler", "heun", "rk4"])
    p.add_argument("--torch-device", default="cpu")
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "croce_validation" / "cache"))
    p.add_argument("--start-subject", type=int, default=1)
    p.add_argument("--end-subject", type=int, default=29)
    p.add_argument("--resume", action="store_true",
                    help="Skip subjects with existing cache files")
    return p.parse_args()


def subject_cache_exists(output_dir: Path, subject_id: int) -> bool:
    cache_file = output_dir / f"subject{subject_id}_cache.npz"
    manifest_file = output_dir / f"subject{subject_id}_manifest.json"
    return cache_file.exists() and manifest_file.exists()


def run_subject(args: argparse.Namespace, subject_id: int, output_dir: Path) -> Dict[str, Any]:
    """Run cache generation for a single subject. Returns timing + result info."""
    subject_output_dir = output_dir / f"subject_{subject_id}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(CACHE_SCRIPT),
        "--data-root", args.data_root,
        "--subject-id", str(subject_id),
        "--segment-start-s", str(args.segment_start_s),
        "--segment-duration-s", str(args.segment_duration_s),
        "--num-particles", str(args.num_particles),
        "--sigma-prop", str(args.sigma_prop),
        "--sigma-nirs", str(args.sigma_nirs),
        "--seed", str(args.seed),
        "--threads", str(args.threads),
        "--parallel-workers", str(args.parallel_workers),
        "--solver-kernel", args.solver_kernel,
        "--torch-device", args.torch_device,
        "--output-dir", str(subject_output_dir),
    ]
    if args.use_artifact_eeg:
        cmd.append("--use-artifact-eeg")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    wall_s = time.perf_counter() - t0

    success = result.returncode == 0
    info: Dict[str, Any] = {
        "subject_id": subject_id,
        "success": success,
        "wall_s": round(wall_s, 1),
        "returncode": result.returncode,
    }

    if success:
        # Parse manifest for detailed timing
        manifest_path = subject_output_dir / "cache_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                info["anchors"] = manifest.get("anchors_processed", 0)
                info["total_wall_s"] = manifest.get("timing", {}).get("total_wall_s", 0)
                info["cache_size_mb"] = manifest.get("cache_size_mb", 0)
                info["avg_pf_per_anchor"] = manifest.get("timing", {}).get("avg_pf_per_anchor_s", 0)
            except (json.JSONDecodeError, KeyError):
                pass
        # Copy cache and manifest to top-level output dir
        for fname in subject_output_dir.iterdir():
            if fname.suffix in (".npz", ".json"):
                target = output_dir / fname.name
                if not target.exists():
                    target.write_bytes(fname.read_bytes())
    else:
        info["stderr_tail"] = result.stderr[-500:] if result.stderr else ""

    return info


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(range(args.start_subject, args.end_subject + 1))
    skipped = 0
    if args.resume:
        new_subjects = []
        for sid in subjects:
            if subject_cache_exists(output_dir, sid):
                skipped += 1
            else:
                new_subjects.append(sid)
        if skipped:
            print(f"Skipping {skipped} subjects with existing cache files")
        subjects = new_subjects

    print(f"{'='*72}")
    print(f"BATCH TARGET CACHE GENERATION")
    print(f"{'='*72}")
    print(f"Subjects: {args.start_subject}–{args.end_subject} ({len(subjects)} to process)")
    print(f"Config: sp={args.sigma_prop}, sn={args.sigma_nirs}, N={args.num_particles}")
    print(f"Solver: {args.solver_kernel}, Workers: {args.parallel_workers}, Threads: {args.threads}")
    print(f"Segment: {args.segment_start_s}s–{args.segment_start_s + args.segment_duration_s}s")
    print(f"Output: {output_dir}")
    print()

    all_results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    success_count = 0
    fail_count = 0

    for i, sid in enumerate(subjects):
        print(f"\n--- Subject {sid} ({i+1}/{len(subjects)}) ---")
        info = run_subject(args, sid, output_dir)
        all_results.append(info)

        status = "OK" if info["success"] else "FAIL"
        if info["success"]:
            success_count += 1
            anchors = info.get("anchors", "?")
            wall = info.get("total_wall_s", info["wall_s"])
            size = info.get("cache_size_mb", "?")
            print(f"  {status}: {anchors} anchors, {wall:.0f}s, {size:.1f}MB")
        else:
            fail_count += 1
            print(f"  {status}: {info.get('stderr_tail', 'unknown error')[:200]}")

    total_wall = time.perf_counter() - total_start

    # Save batch manifest
    batch_manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "solver_kernel": args.solver_kernel,
            "parallel_workers": args.parallel_workers,
            "threads": args.threads,
            "num_particles": args.num_particles,
            "sigma_prop": args.sigma_prop,
            "sigma_nirs": args.sigma_nirs,
            "segment_start_s": args.segment_start_s,
            "segment_duration_s": args.segment_duration_s,
            "subjects_start": args.start_subject,
            "subjects_end": args.end_subject,
        },
        "summary": {
            "total_wall_s": round(total_wall, 1),
            "total_wall_h": round(total_wall / 3600, 2),
            "subjects_attempted": len(subjects) + skipped,
            "subjects_processed": len(subjects),
            "subjects_skipped": skipped,
            "success": success_count,
            "failed": fail_count,
        },
        "per_subject": all_results,
    }
    manifest_path = output_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"BATCH COMPLETE")
    print(f"{'='*72}")
    print(f"Total wall time: {total_wall:.0f}s = {total_wall/60:.1f}min = {total_wall/3600:.2f}h")
    print(f"Success: {success_count}, Failed: {fail_count}, Skipped: {skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
