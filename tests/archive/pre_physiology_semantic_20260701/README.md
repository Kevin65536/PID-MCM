# Pre-physiology-semantic compatibility tests

_Frozen old-contract regression tests, 2026-07-02_

---

## 🧪 Collection policy

These tests cover the archived source/observation model, coupling losses, suite generators, and scorecards. `pytest.ini` excludes `tests/archive/` from default collection.

Run an exact archived file only when changing the compatibility surface. New-architecture correctness tests must live in the active `tests/` tree and must not import `src.compatibility`.

_Last updated: 2026-07-02_
