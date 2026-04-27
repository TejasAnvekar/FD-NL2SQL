final_results bundle

Files:
- all_methods_table_metrics.csv/json: raw every-method table with numeric Table F1, ROUGE-L F1, and CHRF
- all_methods_table_metrics_formatted.csv/json/md: presentation-ready versions with percentages rounded to 2 decimals; bold marks the best metric within each method family among displayed rows; same-model planner/executor pipelines are excluded here
- run_dirs/: symlinks to the source run directories for every method in the main table, including TableGPT
- ablation_studies/same_planner_reasoner_metrics.csv/json: raw same-model planner/executor ablation table
- ablation_studies/same_planner_reasoner_metrics_formatted.csv/json/md: presentation-ready same-model planner/executor ablation table
- ablation_studies/planner_coder_model_metrics.csv/json: raw planner-coder model ablation table
- ablation_studies/planner_coder_model_metrics_formatted.csv/json/md: presentation-ready planner-coder ablation table
- ablation_studies/run_dirs/: symlinks to ablation-study run directories
