# BIOT adapter-alignment v2 status

BIOT remains the only active new comparison method. Its six supported
classification cells have completed full-public adapter replay and pass A0–A7.
REFED regression is preregistered as unsupported because the frozen BIOT
adapter cannot preserve its semantically real partial terminal time support.
No protected manifest or array was opened.

| Task | Unique public samples | Disposition |
| --- | ---: | --- |
| Motor imagery | 1,740 | A0–A7 pass; A8 pending |
| Mental arithmetic | 1,740 | A0–A7 pass; A8 pending |
| WG | 1,560 | A0–A7 pass; A8 pending |
| N-back | 702 | A0–A7 pass; A8 pending |
| DSR | 8,980 | A0–A7 pass; A8 pending |
| Visual | 7,720 | A0–A7 pass; A8 pending |
| REFED regression | — | unsupported: `BIOT_NO_PARTIAL_TIME_MASK_CONTRACT` |

The retained machine evidence is in [`evidence/alignment_v2`](evidence/alignment_v2),
and can be regenerated with:

```bash
.venv/bin/python comparative_methods/BIOT/audit_alignment_v2.py
```

A8 is deliberately still pending. It requires a frozen public-development job
matrix, retry/failure rules, and a separate protected unlock decision. The
serial queue therefore remains on BIOT; CBraMod implementation or execution
must not start yet.
