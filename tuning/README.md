# Hyperparameter tuning (optional)

This folder is **not** used for inference. It contains the Optuna search that originally produced the files in `tuning_results_5foldcv_2000_iter_ms_augmented/`.

**Requirements:** `pip install optuna`, training data in `data/df_plus_ms.dat` and `data/classes_plus_ms.dat`.

**Run from repository root:**

```bash
python -m tuning.tuning_script
```

Outputs `output_<Package>.txt` in the current directory (move into `tuning_results_*` if you adopt new best params).
