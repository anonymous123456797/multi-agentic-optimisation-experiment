# ssbse2026

## Content
 - difficulty_analysis_*: breakdowns of pass@1 by problem difficulty for the Pareto fronts
 - error_trends: analysis of the error types for generated solutions, for run 1
 - MBPP: the problem data sets used in the experiments (mbpp_subset_50 and mbpp_with_difficulty)
 - results/population_logs_run5*.zip: the generational data for NSGAIII on the 5 repeat runs using the MBPP 50 subset
 - results/run*_pareto.zip: the rerun of the final Pareto front for each of the 5 runs, using the full MBPP data set
 - results/pareto_front_wth_difficulty_columns_*.pdf: the bar plots showing relationships between hyperparameters and the objectives
 - strat_sample: scripts and data to generate the 50 subset
 - test/unittest.py: needed to test the generated programs
 - seed_configs.json: the seed hyperparameter configurations for the search

## Setup 
1. Install ollama from here: https://ollama.com/
2. Run `bash ollama_pulls.sh` to get the required models
3. Run `ollama run qwen2.5-coder:7b-base-q8_0` to optionally check a model
4. Run `python3 -m pip install -r requirements.txt`
5. Run `python3 test.py` to test the setup

# Experiment
1. Run `python stratified_sample.py mbpp_with_difficulty.jsonl mbpp_subset_25.json --fraction 0.025 --seed 10`
2. Run `python pymoo_problem.py`   - this will write out the populations to population_logs/

If ollama runs on a different url/port, use this:
`export OLLAMA_BASE_URL=http://localhost:21434`

# Results and analysis
1. rerun_pareto_full.py will rerun with the full MBPP: `python rerun_pareto_full.py --population-json population_logs/20260404_124230_gen_075.json --mbpp-file MBPP/mbpp_with_difficulty.jsonl` (will write reruns to a new dir pareto_full_eval_{timestamp})
2. Determin breakdown of pass@1 for each problem difficulty with `python analyse_difficulty_performance_full.py   --eval-dir results/run1_pareto_full_eval_20260330_152850   --mbpp MBPP/mbpp_with_difficulty.jsonl   --out-dir difficulty_analysis_1`
2. Generate bar plots of Pareto front with `python plot_hparams_along_front_vector.py --latest --folder population_logs --output pareto_front_columns.png`
   or `python plot_hparams_along_front_vector.py --file population_logs/20260328_173159_gen_010.json --output gen10_columns.png`  (change png to pdf as needed)
3. Generate bar plots of Pareto front using results from full MBPP reruns with `python plot_hparams_along_rerun_front_vector.py --csv results/run5_pareto_full_eval_20260405_100149/pareto_full_summary5.csv --output results/pareto_hparams_run5_fullmbpp.png`
4. Generate bar plots of Pareto front using results from full MBPP reruns, adding breakdown of problem difficulty, with `python plot_front_with_difficulty_columns_vector.py   --front-csv results/run1_pareto_full_eval_20260330_152850/pareto_full_summary1.csv   --difficulty-csv difficulty_analysis/difficulty_per_config.csv   --output results/pareto_front_with_difficulty_columns_1.pdf   --sort runtime` (plot used in paper)

## Models

| Model | Size            | Ollama Tag |
| --- |-----------------| --- |
| Qwen2.5-Coder | 1.5B / 7B / 32B | `ollama run qwen2.5-coder` |

## Troubleshooting
- To check if the model is running, run the following `curl http://localhost:11434/api/tags`

## Useful Links
- [Ollama](https://ollama.com/)
- [NSGA-III](https://pymoo.org/algorithms/moo/nsga3.html)
- [PyMoo](https://pymoo.org/)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
