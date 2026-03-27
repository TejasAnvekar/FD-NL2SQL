# Phase Summary

| Phase | Range | Pred w/ SQL | Pred exec ok | Exec match rate | Avg F1 | Acceptance | Seed rows added |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0-499 | 500/500 | 500/500 | 0.1020 | 0.0653 | 68/1500 | 63 |
| 2 | 500-999 | 500/500 | 500/500 | 0.0940 | 0.0530 | 64/1500 | 60 |
| 3 | 1000-1499 | 500/500 | 500/500 | 0.1140 | 0.0618 | 78/1500 | 73 |

## Trend Notes

- Phase 1 -> 2: exec match rate 0.1020 -> 0.0940, avg F1 0.0653 -> 0.0530, accepted 68 -> 64.
- Phase 2 -> 3: exec match rate 0.0940 -> 0.1140, avg F1 0.0530 -> 0.0618, accepted 64 -> 78.
