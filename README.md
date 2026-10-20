# brain-organoid-mea-ml

Code, input data, executed notebooks, and results for the MEA analyses in Figures 3–5 of *Interpretable baseline-aware machine learning for analyzing pharmacological responses in human cortical organoids*.

## Run

Use Python 3.10 or 3.11. From this directory:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `notebooks/` and run all cells in notebooks 1–7 in order. The input is `data/paired_mea_features_raw.csv`; outputs are saved in `results/`. Figures are saved as individual PNG files.

| Notebook | Analysis |
| --- | --- |
| [1_prepare_dataset.ipynb](notebooks/1_prepare_dataset.ipynb) | Validate input and construct feature matrices |
| [2_descriptive_figure3.ipynb](notebooks/2_descriptive_figure3.ipynb) | Figure 3 |
| [3_train_models.ipynb](notebooks/3_train_models.ipynb) | Compare 21 model–representation combinations |
| [4_performance_figure4.ipynb](notebooks/4_performance_figure4.ipynb) | Figure 4 performance plots |
| [5_interpret_model.ipynb](notebooks/5_interpret_model.ipynb) | Held-out SHAP values |
| [6_feature_family_analysis.ipynb](notebooks/6_feature_family_analysis.ipynb) | Family-removal and single-family models |
| [7_interpretation_figure5.ipynb](notebooks/7_interpretation_figure5.ipynb) | Figure 5 |

## Analysis

The input contains 90 paired wells in 15 recording groups and 44 MEA features. Undefined AxIS values are filled with zero. Mean = (Base + Treat) / 2; Delta = Treat − Base. Mean + Delta, Delta-only, and MFR representations contain 88, 44, and 2 variables.

Evaluation uses five grouped outer folds repeated three times, four grouped inner folds, and 24 search trials per fit, with training seed 0. Preprocessing is fitted within training folds. OOF probabilities are averaged across repeats; classification uses a threshold of 0.5. Metric intervals use 2,000 recording-group bootstrap draws; ROC/PR bands use 1,000 draws. SHAP uses ten permutation cycles with the corresponding outer-training background.

The included results select XGBoost with Mean + Delta: pooled OOF AUROC 0.909 and AUPRC 0.953.
