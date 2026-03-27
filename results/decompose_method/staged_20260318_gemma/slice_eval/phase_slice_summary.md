# Phase Slice Evaluation Summary

| Phase | Range | Pred w/ SQL | Exec OK | Exec Match | Exec Match Rate | Avg F1 | Avg SQL AST |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0-499 | 500/500 | 500/500 | 212/500 | 0.4240 | 0.6885 | 0.8491 |
| 2 | 500-999 | 500/500 | 500/500 | 130/500 | 0.2600 | 0.5064 | 0.7748 |
| 3 | 1000-1499 | 500/500 | 500/500 | 152/500 | 0.3040 | 0.5347 | 0.7838 |

## Aggregate

- Total predicted items: 1500
- Total exec-ready items: 1500
- Total exec exact matches: 494
- Overall exec exact match rate: 0.3293
- Macro avg F1: 0.5765
- Weighted avg F1: 0.5765
- Macro avg SQL AST similarity: 0.8026
- Weighted avg SQL AST similarity: 0.8026
