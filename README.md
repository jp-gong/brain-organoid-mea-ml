# brain-organoid-mea-ml

Code, input data, executed notebooks, and results accompanying the paper *Interpretable baseline-aware machine learning for analyzing pharmacological responses in human cortical organoids*.

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
| [2_descriptive_figure3.ipynb](notebooks/2_descriptive_figure3.ipynb) | Figure 3 descriptive plots |
| [3_train_models.ipynb](notebooks/3_train_models.ipynb) | Compare 21 model–representation combinations |
| [4_performance_figure4.ipynb](notebooks/4_performance_figure4.ipynb) | Figure 4 performance plots |
| [5_interpret_model.ipynb](notebooks/5_interpret_model.ipynb) | Held-out SHAP values |
| [6_feature_family_analysis.ipynb](notebooks/6_feature_family_analysis.ipynb) | Family-removal and single-family models |
| [7_interpretation_figure5.ipynb](notebooks/7_interpretation_figure5.ipynb) | Figure 5 interpretation plots |
