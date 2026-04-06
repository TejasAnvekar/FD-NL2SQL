# BERT Failure Demo: row_51

Question:
In Hodgkin Lymphoma trials, convert 'Original/Follow up' status into a boolean is_follow_up.

Why this example is useful:
- There is exactly 1 relevant ground-truth row.
- BERT misses that row badly in all three views.
- The top retrieved rows are lexically or structurally similar, but they are still wrong trials.

Ground truth:
- NCT: NCT02684292
- PubMed ID: 33721562
- Trial name: Keynote-204
- Source value: Original publication
- Derived answer: {"is_follow_up": false}

BERT metrics:
- column view: best relevant rank = 152, hit@10 = 0.0, top-1 rowid = 3
- row view: best relevant rank = 137, hit@10 = 0.0, top-1 rowid = 12
- tuple view: best relevant rank = 144, hit@10 = 0.0, top-1 rowid = 144

Files in this folder:
- `summary.json`: structured comparison summary
- `question_metrics.csv`: the saved BERT per-view metrics for this question
- `ground_truth_table.csv`: the exported ground truth
- `ground_truth_concise.csv`: shortened ground truth view
- `topk_retrievals.csv`: full BERT ranked retrieval output
- `top5_per_view.csv`: top 5 BERT rows per view
- `table_copy.csv`: the visible table BERT searched over
- `metadata.json`: original question metadata
