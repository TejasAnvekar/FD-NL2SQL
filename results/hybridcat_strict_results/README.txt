hybridcat_strict_results bundle

This bundle contains separate strict predicted-only evaluations for the hybrid-category runs.
Strict formatting means rows are retained only when predicted_answer_json is non-empty.

Files:
- hybridcat_strict_metrics.csv/json: raw strict-eval summary across the selected hybrid-category runs
- hybridcat_strict_metrics_formatted.csv/json/md: formatted strict-eval summary with percentages rounded to 2 decimals and best-in-family values bolded
- run_dirs/: shadow run directories that contain strict tabular_eval_v3 and derived_eval_v1 outputs
- strict_shadow_manifest.json: mapping from source run dirs to the shadow strict-eval run dirs
