# Seed Growth Holdout Summary

## Strategy

- Holdout question JSON: `/mnt/data1/srchowd3/FD-NL2SQL/data/natural_question_1500.json`
- Holdout GT JSON: `/mnt/data1/srchowd3/FD-NL2SQL/data/natural_question_1500.json`
- Holdout range: `1200-1499`
- Seed main-metric only: `True`
- Exclude fallback from seed: `True`
- Seed uses original SQL: `True`
- Max seed additions per stage: `50`

## Holdout Trend

| Pass | Seed Source | Added | Exec Match | Exec Match Rate | Avg F1 | Avg SQL AST |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | seed_working.json | 0 | 100/300 | 0.3333 | 0.5606 | 0.7856 |
| 1 | seed_working.json | 3 | 101/300 | 0.3367 | 0.5622 | 0.7866 |
| 2 | seed_after_batch_1.json | 1 | 101/300 | 0.3367 | 0.5645 | 0.7879 |
| 3 | seed_after_batch_2.json | 1 | 100/300 | 0.3333 | 0.5593 | 0.7855 |

## Seed Filters

| Stage | Accepted | Main Metric | Skip Non-Main | Skip Fallback | Skip No Canonical | Skip No Gain | Selected | New Canonical Pairs | New Templates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 194 | 161 | 33 | 0 | 0 | 156 | 3 | 3 | 1 |
| 2 | 164 | 125 | 39 | 1 | 0 | 123 | 1 | 1 | 0 |
| 3 | 144 | 102 | 42 | 1 | 0 | 100 | 1 | 1 | 1 |
