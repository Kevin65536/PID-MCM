"""Launcher guards for standardized training entrypoints."""

from __future__ import annotations

import os


def require_standard_training_launcher(task: str) -> None:
    """Reject direct training-script execution outside the canonical launcher."""
    if os.environ.get('NEURAL_TOKEN_TRAIN_LAUNCHER') == '1':
        return

    raise RuntimeError(
        'Direct training-script execution is disabled for this repository. '
        f'Use `bash experiments/scripts/launch_training_nohup.sh --task {task} ...` instead.'
    )


__all__ = ['require_standard_training_launcher']