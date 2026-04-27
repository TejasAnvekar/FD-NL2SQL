final_results_v2
================

This bundle aggregates aug1500 evaluation outputs across prompting, planner-executor,
coding-agent planner-coder, TableGPT, BlendSQL, and EHRAgent runs.

Source manifest: /mnt/data1/srchowd3/FD-NL2SQL/methodv2/results_hub_aug1500_20260415/all_methods_summary_predonly_strict_with_merged_ehragent_and_data2_planexec.csv
Total runs included: 28
Method families: blendsql, coding_agent, ehragent, planner_executor, prompting_baseline, tabular_reasoning_baseline

Metrics included:
- table_f1: existing tabular F1 from tabular_eval_v3
- grounded_row_jaccard: grounded IoU over row-aligned predictions
- grounded_fowlkes_mallows: non-F1 grounded metric computed from precision and recall
- rouge_l_f1 and chrf: existing text-overlap metrics
- derived_norm_target_f1: normalized target-field metric from derived_eval_v1
