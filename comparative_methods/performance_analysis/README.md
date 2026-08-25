# Comparator performance-degradation analyses

> **STATUS: STOPPED.** The P0 synthesis is complete and retained as exploratory
> evidence. Unexecuted mechanism follow-ups are **ABANDONED**; this directory is
> an evidence index, not a current analysis queue or launch entrypoint.

This package contains post-campaign mechanism analyses defined by
`docs/comparisons/PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md`.
The completed first-round synthesis is
`docs/comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`; generated
evidence bundles are indexed under
`comparative_methods/runs/performance_analysis/20260816_p0/`.

Development and model-selection analyses must use public development data or
outer-train-only resampling. Existing protected aggregates may be used only for
the frozen descriptive global result surface. No script in this package may
retune a method against protected predictions or test labels.

Generated artifacts belong under
`comparative_methods/runs/performance_analysis/20260816_p0/` and must retain
source locators, configuration, seeds, sample/subject support, uncertainty
definitions, and an explicit exploratory/confirmatory label.
