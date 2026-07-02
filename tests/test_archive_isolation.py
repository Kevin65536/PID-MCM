from pathlib import Path

from src.tokenizers import list_tokenizers
from src.tokenizers.registry import _TOKENIZER_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_TOKENIZER_NAMES = {
    'factorized_labram_vqnsp',
    'source_observation_labram_vqnsp',
}


def test_default_tokenizer_registry_excludes_pre_redesign_models():
    assert LEGACY_TOKENIZER_NAMES.isdisjoint(list_tokenizers())


def test_legacy_tokenizer_registration_requires_explicit_opt_in():
    from src.compatibility.pre_physiology_semantic_20260701 import register_legacy_tokenizers

    original = dict(_TOKENIZER_REGISTRY)
    try:
        register_legacy_tokenizers()
        assert LEGACY_TOKENIZER_NAMES.issubset(list_tokenizers())
    finally:
        _TOKENIZER_REGISTRY.clear()
        _TOKENIZER_REGISTRY.update(original)


def test_active_python_surfaces_do_not_import_compatibility_package():
    candidates = [
        *(
            path
            for path in (PROJECT_ROOT / 'src').rglob('*.py')
            if 'compatibility' not in path.parts
        ),
        *(
            path
            for path in (PROJECT_ROOT / 'experiments' / 'scripts').glob('*.py')
        ),
    ]
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in candidates
        if 'src.compatibility' in path.read_text(encoding='utf-8')
    ]
    assert offenders == []


def test_active_config_root_contains_only_target_and_archive_namespaces():
    config_root = PROJECT_ROOT / 'experiments' / 'configs'
    names = {path.name for path in config_root.iterdir() if path.is_dir()}
    assert names == {'archive', 'physiology_semantic_tokenizer'}
