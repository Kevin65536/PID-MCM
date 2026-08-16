#!/usr/bin/env python3
"""Build a read-only source-anchor readiness inventory.

This command deliberately does not download data, call a remote API, import a
method, or launch training.  It records what can be established from the
pinned local source checkout, local weights, and the method-level provenance
documents.  A row is *not* an assertion that the source paper's number is
reproduced: ``anchor_status`` is ``not_verifiable`` whenever a released data
split, checkpoint, head, or preprocessing contract is missing.

Example
-------
    python comparative_methods/performance_analysis/source_anchor_inventory.py

The default output is the P0 performance-analysis run directory.  Use
``--output-dir`` to write an isolated inventory elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "comparative_methods"
    / "runs"
    / "performance_analysis"
    / "20260816_p0"
    / "source_anchors"
)


def _p(relative: str) -> Path:
    return REPO_ROOT / relative


def _paths_exist(paths: Iterable[str]) -> bool:
    return all(_p(path).exists() for path in paths)


def _git_head(relative: str) -> str | None:
    path = _p(relative)
    if not path.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _record(
    *,
    method: str,
    method_id: str,
    official_code_url: str,
    official_code_revision: str | None,
    local_code_path: str,
    license_status: str,
    official_weights_status: str,
    official_weights_local_paths: list[str],
    official_data_status: str,
    source_anchor_tasks: str,
    exact_task_verifiability: str,
    split_verifiability: str,
    metric_verifiability: str,
    head_verifiability: str,
    preprocess_verifiability: str,
    source_fidelity_boundary: str,
    anchor_status: str,
    blocking_items: list[str],
    runnable_command_or_not_verifiable: str,
    evidence_paths: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "method_id": method_id,
        "official_code_url": official_code_url,
        "official_code_revision": official_code_revision,
        "local_code_path": local_code_path,
        "official_code_available": _p(local_code_path).is_dir(),
        "license_status": license_status,
        "official_weights_status": official_weights_status,
        "official_weights_local_paths": official_weights_local_paths,
        "official_weights_available": _paths_exist(official_weights_local_paths)
        if official_weights_local_paths
        else False,
        "official_data_status": official_data_status,
        "source_anchor_tasks": source_anchor_tasks,
        "exact_task_verifiability": exact_task_verifiability,
        "split_verifiability": split_verifiability,
        "metric_verifiability": metric_verifiability,
        "head_verifiability": head_verifiability,
        "preprocess_verifiability": preprocess_verifiability,
        "source_fidelity_boundary": source_fidelity_boundary,
        "anchor_status": anchor_status,
        "blocking_items": blocking_items,
        "runnable_command_or_not_verifiable": runnable_command_or_not_verifiable,
        "evidence_paths": evidence_paths,
        "notes": notes,
    }


def build_inventory() -> list[dict[str, Any]]:
    """Return the seven method records, using only local evidence."""

    return [
        _record(
            method="BIOT",
            method_id="biot",
            official_code_url="https://github.com/ycq091044/BIOT",
            official_code_revision="d138e32634e52ae9fa6ec98ac9c4087b14ca869a",
            local_code_path="comparative_methods/BIOT/upstream",
            license_status="MIT; upstream/LICENSE present",
            official_weights_status="official checkpoints bundled in upstream checkout",
            official_weights_local_paths=[
                "comparative_methods/BIOT/upstream/pretrained-models/EEG-PREST-16-channels.ckpt",
                "comparative_methods/BIOT/upstream/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt",
                "comparative_methods/BIOT/upstream/pretrained-models/EEG-six-datasets-18-channels.ckpt",
            ],
            official_data_status=(
                "preprocessing/source dataset entrypoints are bundled, but TUAB/TUEV/"
                "CHB-MIT/SHHS data are not bundled"
            ),
            source_anchor_tasks="TUAB binary; TUEV multiclass",
            exact_task_verifiability="yes for upstream task entrypoints and README command surfaces",
            split_verifiability=(
                "partial: source loaders expose train/eval/test, but the released local "
                "checkout has no data manifest for reconstructing the paper table split"
            ),
            metric_verifiability=(
                "partial: README reports TUAB balanced accuracy/AUC and TUEV balanced "
                "accuracy/Kappa/weighted F1; exact table aggregation requires source data"
            ),
            head_verifiability="yes: official BIOT supervised classifier and end-to-end path are present",
            preprocess_verifiability=(
                "yes for source 200 Hz, 200-sample token, 100-sample hop and native montage "
                "entrypoints; target-domain native-electrode transfer is separate"
            ),
            source_fidelity_boundary="official BIOT encoder/checkpoint; source task head and data only for anchor",
            anchor_status="conditionally_runnable_missing_external_data",
            blocking_items=[
                "required upstream datasets and their subject/file manifests are absent",
                "paper table training environment and exact split provenance are not locally frozen",
                "current benchmark adapter uses native 16 measured electrodes and frozen linear head, not the source bipolar/end-to-end path",
            ],
            runnable_command_or_not_verifiable=(
                "cd comparative_methods/BIOT/upstream && python run_binary_supervised.py "
                "--dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 "
                "--hop_length 100 --sample_length 10 --batch_size 512 --model BIOT "
                "--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt; "
                "not_verifiable until upstream data are supplied"
            ),
            evidence_paths=[
                "comparative_methods/BIOT/sources/method_manifest.yaml",
                "comparative_methods/BIOT/sources/SOURCE_FIDELITY.md",
                "comparative_methods/BIOT/upstream/README.md",
                "comparative_methods/BIOT/upstream/run_binary_supervised.py",
                "comparative_methods/BIOT/upstream/run_multiclass_supervised.py",
            ],
            notes=(
                "A source anchor is technically plausible after data provisioning; it must not be "
                "called a reproduction until source split and paper metric aggregation are verified."
            ),
        ),
        _record(
            method="CBraMod",
            method_id="cbramod",
            official_code_url="https://github.com/wjq-learning/CBraMod",
            official_code_revision="0ff6be918985689e7df679bc731ffb70e6c6224f",
            local_code_path="comparative_methods/CBraMod/upstream",
            license_status="MIT; upstream/LICENSE present",
            official_weights_status="official Hugging Face checkpoint bundled locally",
            official_weights_local_paths=[
                "comparative_methods/CBraMod/checkpoints/pretrained_weights.pth",
            ],
            official_data_status=(
                "dataset loaders and preprocessing scripts are bundled; selected downstream data "
                "are not bundled"
            ),
            source_anchor_tasks=(
                "FACED; SEED-V; PhysioNet-MI; SHU-MI; ISRUC; CHB-MIT; BCIC2020-3; "
                "Mumtaz2016; SEED-VIG; MentalArithmetic; TUEV; TUAB; BCIC-IV-2a"
            ),
            exact_task_verifiability="yes for named official downstream loader/config entrypoints",
            split_verifiability=(
                "partial: loaders consume train/val/test directories; exact subject-disjoint "
                "membership is not packaged as a local manifest"
            ),
            metric_verifiability=(
                "yes for source evaluator (balanced accuracy, Kappa, weighted F1, binary AUROC/PR-AUC, regression); "
                "paper table aggregation still needs data"
            ),
            head_verifiability=(
                "yes: official finetune_main exposes all-patch, one/two/three-layer and average-pool heads; "
                "quick_example is executable with a mock tensor"
            ),
            preprocess_verifiability=(
                "yes for 200 Hz/200-sample patch and source dataset preprocessing; current shared adapter "
                "mean-token linear probe is a documented transfer boundary"
            ),
            source_fidelity_boundary="official CBraMod encoder + official pretrained_weights; source downstream head for anchor",
            anchor_status="conditionally_runnable_missing_external_data",
            blocking_items=[
                "official downstream datasets and exact split manifests are absent",
                "source paper's full-fine-tune selection/configuration is not a single immutable local manifest",
                "current benchmark uses frozen mean-token linear probe; source paper emphasizes full fine-tuning",
            ],
            runnable_command_or_not_verifiable=(
                "cd comparative_methods/CBraMod/upstream && python finetune_main.py "
                "--downstream_dataset MentalArithmetic --datasets_dir /path/to/processed "
                "--use_pretrained_weights True --foundation_dir "
                "../checkpoints/pretrained_weights.pth; "
                "not_verifiable until data and source split manifests are supplied"
            ),
            evidence_paths=[
                "comparative_methods/CBraMod/sources/method_manifest.yaml",
                "comparative_methods/CBraMod/sources/SOURCE_FIDELITY.md",
                "comparative_methods/CBraMod/REPRESENTATION_LAYER_AUDIT.md",
                "comparative_methods/CBraMod/upstream/README.md",
                "comparative_methods/CBraMod/upstream/finetune_main.py",
                "comparative_methods/CBraMod/upstream/finetune_evaluator.py",
            ],
            notes=(
                "CBraMod is the most actionable source anchor for the adapter-capacity question: "
                "the source code supports official full fine-tuning, while the shared benchmark freezes it."
            ),
        ),
        _record(
            method="REVE",
            method_id="reve",
            official_code_url="https://github.com/elouayas/reve_eeg",
            official_code_revision="06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7",
            local_code_path="comparative_methods/REVE/upstream",
            license_status="MIT; upstream/LICENSE present; model weights use gated responsible-use terms",
            official_weights_status="official base/large encoder and position-bank assets downloaded under gated terms",
            official_weights_local_paths=[
                "comparative_methods/REVE/checkpoints/reve-base/model.safetensors",
                "comparative_methods/REVE/checkpoints/reve-large/model.safetensors",
                "comparative_methods/REVE/checkpoints/reve-positions/model.safetensors",
            ],
            official_data_status=(
                "dataset-specific preprocessing scripts and LMDB split format are bundled; most downstream "
                "data are not bundled; partial HF dataset is advertised"
            ),
            source_anchor_tasks=(
                "BCIC-IV-2a; FACED; HMC; ISRUC; MUMTAZ; PhysioNet-MI; Speech/BCIC2020; "
                "Stress/MentalArithmetic; TUAB; TUEV"
            ),
            exact_task_verifiability="yes for task configs, loader names and LP/FT entrypoints",
            split_verifiability=(
                "partial-to-yes: LMDB train/val/test membership is executable when prepared data exist, "
                "but the original paper's complete per-dataset membership is not locally bundled"
            ),
            metric_verifiability=(
                "yes for official evaluator (accuracy, balanced accuracy, Kappa, weighted F1, AUROC, PR-AUC); "
                "source-paper aggregation and overlap policy remain separate"
            ),
            head_verifiability=(
                "partial: official LP leaves cls_query_token trainable and supports FT; current shared adapter "
                "freezes query and trains only a linear head"
            ),
            preprocess_verifiability=(
                "yes for official 200 Hz, patch/overlap and position-bank route; exact task-specific scale factors "
                "are config-dependent and must be retained for anchor"
            ),
            source_fidelity_boundary="official REVE encoder/position bank; source LP or LP+FT head policy for anchor",
            anchor_status="conditionally_runnable_missing_external_data",
            blocking_items=[
                "gated model terms prohibit redistribution and require the accepted local access path",
                "official downstream data and exact paper split manifests are not locally complete",
                "current support-matched route freezes cls_query_token, unlike source LP",
                "Shin2017A is declared pretraining-overlapping, so Single-Trial anchor cannot be target-excluded",
            ],
            runnable_command_or_not_verifiable=(
                "cd comparative_methods/REVE/upstream && torchrun --nproc_per_node=1 src/dt.py "
                "task=physio training_mode=lp data_root=/path/to/datasets "
                "pretrained_path=/local/path/reve-base; not_verifiable until prepared LMDB data, "
                "accepted gated access and task config are supplied"
            ),
            evidence_paths=[
                "comparative_methods/REVE/sources/method_manifest.yaml",
                "comparative_methods/REVE/sources/SOURCE_FIDELITY.md",
                "comparative_methods/REVE/IDENTITY_AND_REPRESENTATION_AUDIT.md",
                "comparative_methods/REVE/upstream/README.md",
                "comparative_methods/REVE/upstream/src/dt.py",
                "comparative_methods/REVE/upstream/src/eval_dt.py",
                "comparative_methods/REVE/checkpoints/reve-base/README.md",
            ],
            notes=(
                "The official model-card mock forward is locally inspectable, but a numerical paper anchor still "
                "requires the matching prepared data and declared trainable-query policy."
            ),
        ),
        _record(
            method="NormWear",
            method_id="normwear_eeg_fnirs_adapted",
            official_code_url="https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear",
            official_code_revision="07517fcb13def8c89cb586128359cec02f86ec8d",
            local_code_path="comparative_methods/NormWear/upstream",
            license_status="Apache-2.0 root license; modules/normwear.py retains an additional notice",
            official_weights_status="official release backbone and optional MSiTF checkpoint downloaded locally",
            official_weights_local_paths=[
                "comparative_methods/NormWear/checkpoints/normwear_pretrain_ckpt.pth",
                "comparative_methods/NormWear/checkpoints/normwear_msitf_zeroshot_last_checkpoint-5.pth",
            ],
            official_data_status=(
                "official pretraining/downstream data are external downloads; README documents a local "
                "pickle + train_test_split.json format"
            ),
            source_anchor_tasks="11 wearable datasets / 18 applications in paper; no released fNIRS anchor task",
            exact_task_verifiability=(
                "partial: official downstream group/task registry is bundled, but target project EEG-fNIRS "
                "adaptation is not an original NormWear task"
            ),
            split_verifiability=(
                "partial: source consumes train_test_split.json and uid; exact paper subject policy and files "
                "are external"
            ),
            metric_verifiability=(
                "yes for classification helper (accuracy, precision, recall, F1, ROC-AUC/AP); exact paper table "
                "aggregation still needs the released datasets"
            ),
            head_verifiability=(
                "yes for source linear-probe and fine-tune entrypoints; current benchmark executes only the "
                "adapted encoder plus outer-training linear probe"
            ),
            preprocess_verifiability=(
                "yes for source CWT/token/65 Hz executable model; target fNIRS requires explicit EEG/HbO/HbR "
                "identity and 200/10-to-65 Hz adaptation"
            ),
            source_fidelity_boundary="official NormWear encoder checkpoint; `normwear_eeg_fnirs_adapted` for target modality",
            anchor_status="not_verifiable_original_paper_anchor",
            blocking_items=[
                "fNIRS is not among the declared pretraining modalities and no original-paper fNIRS task is released",
                "official processed data are external and not present in checkout",
                "paper source split membership/subject stratification cannot be reconstructed from local files alone",
                "an EEG-fNIRS target anchor would be an adaptation, not a NormWear paper reproduction",
            ],
            runnable_command_or_not_verifiable=(
                "CUDA_VISIBLE_DEVICES=0 python -m NormWear.downstream_main --model_name normwear "
                "--model_weight_dir comparative_methods/NormWear/checkpoints/normwear_pretrain_ckpt.pth "
                "--group 0 --data_path /path/to/processed_data --num_runs 1 --prepare_embed 1; "
                "not_verifiable for the current EEG-fNIRS benchmark because the source task data are absent "
                "and fNIRS is an explicit adaptation"
            ),
            evidence_paths=[
                "comparative_methods/NormWear/sources/method_manifest.yaml",
                "comparative_methods/NormWear/sources/SOURCE_FIDELITY.md",
                "comparative_methods/NormWear/IDENTITY_AND_ADAPTATION_AUDIT.md",
                "comparative_methods/NormWear/upstream/README.md",
                "comparative_methods/NormWear/upstream/main_model.py",
                "comparative_methods/NormWear/upstream/downstream_pipeline/linear_prob_main.py",
            ],
            notes=(
                "A source-compatible EEG wearable anchor is possible after data provisioning; it cannot validate "
                "the adapted fNIRS comparison against an original fNIRS claim."
            ),
        ),
        _record(
            method="EFRM",
            method_id="efrm_a_multimodal_eeg_fnirs_representation_learning_model",
            official_code_url="https://github.com/EuijinMisp/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model",
            official_code_revision="a62bf3d4c092ac3022b6c0bad90ec3993d5a5720",
            local_code_path="comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model",
            license_status="no LICENSE file observed; redistribution/terms must be clarified",
            official_weights_status="no official released pretrained checkpoint observed in the source checkout",
            official_weights_local_paths=[],
            official_data_status=(
                "README specifies user-supplied .npy arrays and folder layouts; no source pretraining or "
                "downstream data are bundled"
            ),
            source_anchor_tasks="sleepstage EEG; mental-arithmetic fNIRS; drowsiness EEG/fNIRS/multimodal",
            exact_task_verifiability="yes for source CLI task names and dataloader classes",
            split_verifiability=(
                "not_verifiable: code chooses deterministic lists/k-shot slices but does not provide a released "
                "paper subject/session manifest or exact source membership"
            ),
            metric_verifiability=(
                "partial: source utils computes macro precision/recall/F1 and mean; original paper task-table "
                "aggregation and few-shot repetition policy are not fully encoded"
            ),
            head_verifiability=(
                "yes for source EEG/fNIRS/paired ViT wrappers and linear-probe/fine-tune modes; no checkpoint "
                "means source weights cannot be loaded locally"
            ),
            preprocess_verifiability=(
                "partial: source code exposes 24x1024 EEG, 64x128 fNIRS, 0.5 mask and random augmentation/cropping; "
                "paper acquisition preprocessing and target-domain resampling need independent verification"
            ),
            source_fidelity_boundary="official EFRM architecture/CLI only; project EFRM-PyTorch is an independent retraining track",
            anchor_status="not_verifiable_no_official_checkpoint_or_data",
            blocking_items=[
                "no official pretrained checkpoint is bundled or referenced by a verifiable local hash",
                "pretraining and downstream source data are explicitly user-supplied and absent",
                "paper's exact subject/few-shot split and repetition policy are not encoded in the checkout",
                "no license file is present in the local source checkout",
            ],
            runnable_command_or_not_verifiable=(
                "cd comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/EFRM && "
                "python main_linearprobe.py --gpu_ids \"[0]\" --mode linprobe "
                "--target_dataset_type mental_arithmetic_fnirs --k_shot 1 --n_class 2 "
                "--pretrained_model_path /path/to/source_checkpoint; not_verifiable because both "
                "the source checkpoint and required data are absent"
            ),
            evidence_paths=[
                "comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/README.md",
                "comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/EFRM/dataloader.py",
                "comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/EFRM/model_pretrain.py",
                "comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/EFRM/model_finetune.py",
                "comparative_methods/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/EFRM/utils.py",
                "comparative_methods/EFRM-PyTorch/sources/method_manifest.yaml",
            ],
            notes=(
                "Existing EFRM-PyTorch retraining/alignment artifacts are evidence about the project adaptation, "
                "not a source-anchor reproduction of the original EFRM release."
            ),
        ),
        _record(
            method="STA-Net",
            method_id="sta_net_eeg_fnirs_supervised",
            official_code_url="https://github.com/MutianLiu-SHU/STA-Net",
            official_code_revision="b6db8bb5eb2f6491a13f0938880ee70e32162ee7",
            local_code_path="comparative_methods/STA-Net",
            license_status="no license file observed; redistribution is blocked pending clarification",
            official_weights_status="not applicable: source runner trains from random initialization",
            official_weights_local_paths=[],
            official_data_status=(
                "README links two open-access EEG-NIRS datasets; runner expects pre-generated per-subject NPZ "
                "files at a hard-coded Windows path, none bundled"
            ),
            source_anchor_tasks="binary MI, MA, WG; three sessions per subject",
            exact_task_verifiability=(
                "yes for runner task loop and three-session leave-one-session-out source protocol; source task data "
                "must be supplied"
            ),
            split_verifiability=(
                "yes for code-level split mechanics (one 200-trial session held out; 80 of remaining 400 used as "
                "validation with seed 42), but exact source subject list/data files are absent"
            ),
            metric_verifiability=(
                "partial: runner compiles/evaluates Keras accuracy; paper additionally reports Kappa, whose exact "
                "aggregation requires source data"
            ),
            head_verifiability="yes for official TensorFlow FGSA/EGTA/fusion model and Adam/early-stopping loop",
            preprocess_verifiability=(
                "partial: source preprocessing scripts and NPZ contract exist, but the complete preprocessing "
                "provenance and raw-data conversion are not packaged as a one-command pipeline"
            ),
            source_fidelity_boundary="official TensorFlow STA-Net runner/model with source per-subject protocol",
            anchor_status="not_verifiable_hard_coded_path_and_external_data",
            blocking_items=[
                "runner hard-codes E:\\IF\\dataset\\model_input and is not portable without a source-preserving path edit",
                "required pre-generated NPZ data are absent",
                "no license file observed in official checkout",
                "current PyTorch comparison is an independent adaptation and must not be called numerical reproduction",
            ],
            runnable_command_or_not_verifiable=(
                "python comparative_methods/STA-Net/run_sta_net.py after supplying the source NPZ directory and "
                "a source-preserving path configuration; currently not_verifiable because the script's path is "
                "hard-coded and data are absent"
            ),
            evidence_paths=[
                "comparative_methods/STA-Net/README.md",
                "comparative_methods/STA-Net/run_sta_net.py",
                "comparative_methods/STA-Net/sta.py",
                "comparative_methods/STA-Net/preprocessing/preprocessing.py",
                "comparative_methods/STA-Net-PyTorch/sources/method_manifest.yaml",
                "comparative_methods/STA-Net-PyTorch/sources/research_sta_net_feasibility_20260718.md",
            ],
            notes=(
                "This is a source-protocol anchor candidate, not a checkpoint anchor. The existing PyTorch run can "
                "support mechanism diagnostics, but source numerical equivalence remains unverified."
            ),
        ),
        _record(
            method="BrainFusion",
            method_id="brainfusion_nvc_csp_stacking_reimplementation",
            official_code_url="https://github.com/lwh-scut/BrainFusion",
            official_code_revision="1d9dcf4026f237efed7f0dd44ba44ef0bf87915b",
            local_code_path="comparative_methods/BrainFusion-NVC-CSP-Stacking/upstream",
            license_status="BSD-3-Clause; upstream/LICENSE present",
            official_weights_status="not applicable: source case is supervised fold-local estimation",
            official_weights_local_paths=[],
            official_data_status=(
                "official checkout has data I/O and preprocessing utilities but no bundled paper-case data; "
                "TU-Berlin data must be obtained separately"
            ),
            source_anchor_tasks="paper MI case: EEG, HbO, HbR, NVC-CSP and stacked ensemble",
            exact_task_verifiability=(
                "partial: paper case is identifiable from the local paper/provenance audit, but public source "
                "checkout does not expose the complete case execution"
            ),
            split_verifiability=(
                "not_verifiable: paper reports within-subject 10-fold CV per participant, but source checkout does "
                "not package the case split/data manifest"
            ),
            metric_verifiability=(
                "partial: paper reports accuracy summary, but the public run path is a simulation placeholder and "
                "does not expose the paper-case estimator/evaluation implementation"
            ),
            head_verifiability=(
                "not_verifiable: public checkout names NVC-CSP/Integrated Model but does not fit the paper-case CSP "
                "or stacking ensemble"
            ),
            preprocess_verifiability=(
                "partial: source EEG/fNIRS preprocessing and NVC HRF implementation are available; exact paper-case "
                "preprocessing plus estimator selection cannot be reconstructed"
            ),
            source_fidelity_boundary="independent fold-local NVC/CSP/stacking reimplementation; component fidelity only",
            anchor_status="not_verifiable_public_case_execution_missing",
            blocking_items=[
                "public checkout's ML run path is explicitly a simulation placeholder",
                "paper-case CSP and stacking implementation are not released at the pinned revision",
                "paper data/split and estimator-selection artifacts are absent",
                "current benchmark's support-matched time-zero window differs from the source hemodynamic context",
            ],
            runnable_command_or_not_verifiable=(
                "python comparative_methods/BrainFusion-NVC-CSP-Stacking/upstream/src/main.py only launches the GUI; "
                "paper-case NVC-CSP stacking anchor is not_verifiable from the public checkout. Use the project's "
                "fold-local reimplementation after separately registering a protocol bridge."
            ),
            evidence_paths=[
                "comparative_methods/BrainFusion-NVC-CSP-Stacking/sources/method_manifest.yaml",
                "comparative_methods/BrainFusion-NVC-CSP-Stacking/sources/SOURCE_FIDELITY.md",
                "comparative_methods/BrainFusion-NVC-CSP-Stacking/OBSERVATION_BUDGET_AUDIT.md",
                "comparative_methods/BrainFusion-NVC-CSP-Stacking/upstream/README.md",
                "comparative_methods/BrainFusion-NVC-CSP-Stacking/upstream/src/BrainFusion/pipeLine/coupling_analysis.py",
            ],
            notes=(
                "The existing project implementation is correctly named a reimplementation. A source anchor would "
                "require authors' missing case code or an explicitly labelled protocol-bridge experiment."
            ),
        ),
    ]


CSV_FIELDS = [
    "method",
    "method_id",
    "official_code_url",
    "official_code_revision",
    "local_code_path",
    "official_code_available",
    "license_status",
    "official_weights_status",
    "official_weights_local_paths",
    "official_weights_available",
    "official_data_status",
    "source_anchor_tasks",
    "exact_task_verifiability",
    "split_verifiability",
    "metric_verifiability",
    "head_verifiability",
    "preprocess_verifiability",
    "source_fidelity_boundary",
    "anchor_status",
    "blocking_items",
    "runnable_command_or_not_verifiable",
    "evidence_paths",
    "notes",
]


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _report(inventory: list[dict[str, Any]], generated_at: str) -> str:
    counts: dict[str, int] = {}
    for row in inventory:
        status = str(row["anchor_status"])
        counts[status] = counts.get(status, 0) + 1

    lines = [
        "# Source-anchor inventory",
        "",
        f"Generated at `{generated_at}` by `comparative_methods/performance_analysis/source_anchor_inventory.py`.",
        "",
        "This is a read-only local provenance audit. It does not download data, run a model, or claim numerical reproduction.",
        "",
        "## Status summary",
        "",
        f"- methods audited: **{len(inventory)}**",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: **{count}**")

    lines.extend(
        [
            "",
            "The distinction between `conditionally_runnable_missing_external_data` and `not_verifiable_*` is deliberate:",
            "the former has an identifiable official code/weight/data entrypoint but needs external data or manifests;",
            "the latter lacks a source-level artifact required to establish the paper anchor (for example EFRM weights,",
            "STA-Net portable data path, or BrainFusion's paper-case CSP/stacking implementation).",
            "",
            "## Method-level decisions",
            "",
            "| Method | Anchor status | Official code / weights | Main blocking item |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in inventory:
        weights = row["official_weights_status"]
        blockers = str(row["blocking_items"][0])
        lines.append(f"| {row['method']} | `{row['anchor_status']}` | {weights} | {blockers} |")

    lines.extend(
        [
            "",
            "## Reproduction guardrails",
            "",
            "1. A command in `inventory.csv` is a source command template only; do not execute it until the listed data, split, license, and checkpoint conditions are frozen.",
            "2. A project adapter or retraining checkpoint is not an official source anchor. In particular, EFRM-PyTorch and STA-Net-PyTorch remain independent comparison tracks.",
            "3. REVE weights are gated and non-redistributable; retain their local hash and responsible-use status in any follow-up run.",
            "4. NormWear's EEG-fNIRS result is an explicit adaptation because fNIRS is absent from its declared pretraining modalities; it cannot establish an original fNIRS paper number.",
            "5. BrainFusion's public GUI launch is not the paper-case NVC-CSP stacking experiment; only the existing fold-local reimplementation is currently executable for project analysis.",
            "",
            "## Evidence",
            "",
            "The machine-readable `capability.json` contains all evidence paths and the resolved local path checks. Relative paths are rooted at the repository root.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_inventory(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory()
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    git_heads = {
        row["method"]: _git_head(row["local_code_path"]) for row in inventory
    }
    capability = {
        "schema": "source_anchor_inventory_v1",
        "generated_at": generated_at,
        "repository_root": str(REPO_ROOT),
        "read_only": True,
        "external_downloads_performed": False,
        "external_commands_run": False,
        "methods": inventory,
        "resolved_local_git_heads": git_heads,
    }
    csv_path = output_dir / "inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in inventory:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})

    json_path = output_dir / "capability.json"
    _write_json(json_path, capability)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(_report(inventory, generated_at), encoding="utf-8")
    return {"inventory": csv_path, "capability": json_path, "report": report_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for inventory.csv, capability.json and REPORT.md",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    paths = write_inventory(output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
