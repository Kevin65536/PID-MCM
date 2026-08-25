import subprocess
from pathlib import Path

from src.tokenizers import list_tokenizers


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_tokenizer_registry_has_one_owner():
    assert list_tokenizers() == ["physiology_semantic"]


def test_local_archive_is_ignored_and_not_an_active_namespace():
    probe = PROJECT_ROOT / "experiments/archive/probe"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(probe)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0
    for path in (
        PROJECT_ROOT / "src/archive",
        PROJECT_ROOT / "tests/archive",
        PROJECT_ROOT / "experiments/configs/archive",
        PROJECT_ROOT / "experiments/scripts/archive",
    ):
        assert not path.exists()
