"""Optuna search for class weights and XGBoost hyperparameters.

Run from repository root (requires optuna):

    python -m tuning.tuning_script

Writes ``output_<Package>.txt`` in the current working directory.
"""
from pathlib import Path
import sys

import optuna
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import explore_commons
from tuning.XGBoost_Weighted import XGBClassifier_w


def main() -> None:
    np.random.seed(1606421)
    idx = pd.IndexSlice

    data_path = REPO_ROOT / "data" / "df_plus_ms.dat"
    labels_path = REPO_ROOT / "data" / "classes_plus_ms.dat"
    df, classes, header = explore_commons.read_data(str(data_path), str(labels_path))

    df, classes = explore_commons.process_data_frame_2(df, classes, header[2:])
    df = explore_commons.merge_duplicate_wavelength_cols(df)
    df = explore_commons.convert_to_galactic_coords(df)

    BasePackage = df.loc[:, idx["Fitted", ["Teff", "Lum"], "Value", :, :, :, :]]
    BiasPackage = df.loc[:, idx["Adopted", ["RA", "Dec", "PMRA", "PMDec", "Distance"], "Value", :, :, :, :]]
    SpectraPackage = pd.concat(
        [
            df.loc[:, idx["Photometry", :, ["Error"], :, :, :, :]],
            df.loc[:, idx[["Model", "Dereddened"], :, "Value", :, :, :, :]],
        ],
        axis=1,
    )
    SpectraPackage = explore_commons.transform_Spectra(df, corrected_error_2=True)
    PhysicsPackage = pd.concat(
        [
            BasePackage,
            df.loc[:, idx["Adopted", ["E(B-V)", "logg", "[Fe/H]"], "Value", :, :, :, :]],
            df.loc[:, idx["Ancillary", "Tspec", "Value", :, :, :, :]],
            SpectraPackage,
        ],
        axis=1,
    )
    FullPackage = pd.concat([BiasPackage, PhysicsPackage], axis=1)

    tune_iter = 2000
    avail_threads = -1

    train_indices, test_indices = explore_commons.custom_train_test_split_2(
        classes, test_size=0.2, min_class_instances=40
    )

    DataPackages = {
        "Full": FullPackage,
        "Physics": PhysicsPackage,
        "Spectra": SpectraPackage,
        "Bias": BiasPackage,
        "Base": BasePackage,
    }

    for dat_pack in DataPackages.items():
        start = timer()

        X_train_full = dat_pack[1].iloc[train_indices]
        X_test = dat_pack[1].iloc[test_indices]
        y_test = classes.iloc[test_indices]
        y_train_full = classes.iloc[train_indices]

        y_train_full, mlb = explore_commons.classes_to_multilabel(
            [i for i in y_train_full["class"].str.split(", ")]
        )
        y_test = mlb.transform([i for i in y_test["class"].str.split(", ")])

        X_train_full.columns = range(X_train_full.shape[1])
        X_test.columns = range(X_test.shape[1])

        def objective(trial, _X, _y):
            n_estimators = trial.suggest_int("n_estimators", high=2000, low=250, step=250)
            w0 = trial.suggest_float("w0", high=1.0, low=0.001, step=0.001)
            w1 = trial.suggest_float("w1", high=1.0, low=0.001, step=0.001)
            w2 = trial.suggest_float("w2", high=1.0, low=0.001, step=0.001)
            w3 = trial.suggest_float("w3", high=1.0, low=0.001, step=0.001)
            w4 = trial.suggest_float("w4", high=1.0, low=0.001, step=0.001)
            w5 = trial.suggest_float("w5", high=1.0, low=0.001, step=0.001)
            w6 = trial.suggest_float("w6", high=1.0, low=0.001, step=0.001)
            w7 = trial.suggest_float("w7", high=1.0, low=0.001, step=0.001)
            w8 = trial.suggest_float("w8", high=1.0, low=0.001, step=0.001)
            eta = trial.suggest_float("eta", high=0.31, low=0.001, step=0.005)
            max_depth = trial.suggest_int("max_depth", high=10, low=3, step=1)
            min_child_weight = trial.suggest_int("min_child_weight", high=20, low=0, step=2)
            subsample = trial.suggest_float("subsample", high=1.0, low=0.2, step=0.1)
            colsample_bytree = trial.suggest_float("colsample_bytree", high=1.0, low=0.3, step=0.1)
            gamma = trial.suggest_float("gamma", high=10.0, low=0.0, step=0.2)

            model = XGBClassifier_w(
                w0,
                w1,
                w2,
                w3,
                w4,
                w5,
                w6,
                w7,
                w8,
                n_estimators=n_estimators,
                eta=eta,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                tree_method="hist",
                random_state=1606421,
                verbosity=0,
            )

            kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=1606421)
            f1_mac_scorer = metrics.make_scorer(metrics.f1_score, average="macro")
            cv_results = model_selection.cross_val_score(model, _X, _y, scoring=f1_mac_scorer, cv=kf)
            return np.min([np.mean(cv_results), np.median(cv_results)])

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train_full, y_train_full),
            n_trials=tune_iter,
            n_jobs=avail_threads,
        )
        trial = study.best_trial

        print(trial.params)
        print(trial.values)
        xgb_best = XGBClassifier_w(**trial.params, verbosity=0)
        xgb_best.fit(X_train_full, y_train_full)
        y_pred = xgb_best.predict(X_test)
        end = timer()
        with open(f"output_{dat_pack[0]}.txt", "w", encoding="utf-8") as f:
            f.write(
                f"""
{trial.params}
{trial.values}
{str(metrics.accuracy_score(y_pred, y_test))}
{str(metrics.f1_score(y_pred, y_test, average='macro'))}
Time taken: {str(end - start)}"""
            )


if __name__ == "__main__":
    main()
