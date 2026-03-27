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
| baseline | seed_working.json | 0 | 100/300 | 0.3333 | 0.5581 | 0.7886 |
| 1 | seed_working.json | 42 | 100/300 | 0.3333 | 0.5597 | 0.7863 |
| 2 | seed_after_batch_1.json | 47 | 101/300 | 0.3367 | 0.5613 | 0.7875 |
| 3 | seed_after_batch_2.json | 44 | 101/300 | 0.3367 | 0.5637 | 0.7871 |
