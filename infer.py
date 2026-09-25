"""Train from tuned hyperparameters, then score a new PySSED .dat file.

Tuning result files store hyperparameters only (not fitted trees). This script
fits a model with those settings, writes artifacts, and runs inference.

Examples
--------
python infer.py train --package Full
python infer.py predict --input path/to/new_stars.dat --output scores.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import explore_commons

SEED = 1606421
WEIGHT_KEYS = [f"w{i}" for i in range(9)]
PACKAGES = ("Base", "Bias", "Spectra", "Physics", "Full")
DEFAULT_TUNING_DIR = Path("tuning_results_5foldcv_2000_iter_ms_augmented")
DEFAULT_TRAIN_DATA = Path("data/df_plus_ms.dat")
DEFAULT_TRAIN_LABELS = Path("data/classes_plus_ms.dat")
DEFAULT_ARTIFACTS = Path("artifacts")
idx = pd.IndexSlice


def load_tuning_params(tuning_dir: Path, package: str) -> dict:
    path = Path(tuning_dir) / f"output_{package}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No tuning file for package {package}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    params = eval(lines[1])
    if not isinstance(params, dict):
        raise ValueError(f"Could not parse hyperparameter dict from {path}")
    return params


def read_pyssed(data_path: Path, labels_path: Path | None = None, num_header_lines: int = 9):
    """Read a PySSED table. Labels are optional for inference-only files."""
    data_path = Path(data_path)
    delimiter = "," if data_path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(
        data_path,
        skiprows=num_header_lines,
        header=None,
        delimiter=delimiter,
        dtype={0: str},
    )
    df[0] = df[0].str.rstrip("_")

    if labels_path is None:
        classes = pd.DataFrame({0: df[0].astype(str).values, 1: ["Unknown"] * len(df)})
    else:
        classes = pd.read_csv(
            labels_path, header=None, delimiter=delimiter, dtype={0: str}
        )
        classes[0] = classes[0].str.replace(" ", "_")

    header = []
    with data_path.open("r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i >= num_header_lines:
                break
            header.append(line.rstrip().split(delimiter))
    return df, classes, header


def build_packages(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = df.loc[:, idx["Fitted", ["Teff", "Lum"], "Value", :, :, :, :]]
    bias = df.loc[:, idx["Adopted", ["RA", "Dec", "PMRA", "PMDec", "Distance"], "Value", :, :, :, :]]
    spectra = explore_commons.transform_Spectra(df, corrected_error_2=True)
    physics = pd.concat(
        [
            base,
            df.loc[:, idx["Adopted", ["E(B-V)", "logg", "[Fe/H]"], "Value", :, :, :, :]],
            df.loc[:, idx["Ancillary", "Tspec", "Value", :, :, :, :]],
            spectra,
        ],
        axis=1,
    )
    full = pd.concat([bias, physics], axis=1)
    return {
        "Base": base,
        "Bias": bias,
        "Spectra": spectra,
        "Physics": physics,
        "Full": full,
    }


def preprocess(data_path: Path, labels_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, classes, header = read_pyssed(data_path, labels_path)
    df, classes = explore_commons.process_data_frame_2(df, classes, header[2:])
    df = explore_commons.merge_duplicate_wavelength_cols(df)
    df = explore_commons.convert_to_galactic_coords(df)
    return df, classes


def flatten_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = range(out.shape[1])
    return out


def probabilities_to_matrix(y_prob, n_samples: int, n_classes: int) -> np.ndarray:
    """Normalize XGBoost/sklearn predict_proba shapes to (n_samples, n_classes)."""
    if isinstance(y_prob, list):
        cols = []
        for item in y_prob:
            arr = np.asarray(item)
            cols.append(arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr.reshape(-1))
        return np.column_stack(cols)
    arr = np.asarray(y_prob)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[-1] == 2 else arr.reshape(n_samples, n_classes)
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        return arr
    raise ValueError(f"Unexpected predict_proba shape: {arr.shape}")


def fill_empty_predictions(y_pred: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    filled = np.array(y_pred, copy=True)
    if filled.ndim == 1:
        filled = filled.reshape(1, -1)
    for i, row in enumerate(filled):
        if row.sum() < 1:
            row[int(np.argmax(y_prob[i]))] = 1
    return filled


def artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    artifacts_dir = Path(artifacts_dir)
    return {
        "model": artifacts_dir / "model.pkl",
        "mlb": artifacts_dir / "label_binarizer.pkl",
        "columns": artifacts_dir / "feature_columns.pkl",
        "meta": artifacts_dir / "meta.pkl",
    }


def save_artifacts(artifacts_dir: Path, model, mlb, columns, meta: dict) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(artifacts_dir)
    pickle.dump(model, paths["model"].open("wb"))
    pickle.dump(mlb, paths["mlb"].open("wb"))
    pickle.dump(columns, paths["columns"].open("wb"))
    pickle.dump(meta, paths["meta"].open("wb"))


def load_artifacts(artifacts_dir: Path):
    paths = artifact_paths(artifacts_dir)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts. Run `python infer.py train` first. Not found:\n  "
            + "\n  ".join(missing)
        )
    model = pickle.load(paths["model"].open("rb"))
    mlb = pickle.load(paths["mlb"].open("rb"))
    columns = pickle.load(paths["columns"].open("rb"))
    meta = pickle.load(paths["meta"].open("rb"))
    return model, mlb, columns, meta


def cmd_train(args: argparse.Namespace) -> None:
    np.random.seed(SEED)
    if not Path(args.data).exists() or not Path(args.labels).exists():
        raise FileNotFoundError(
            "Training tables are not in this clone. Place PySSED files at:\n"
            f"  {args.data}\n  {args.labels}"
        )

    print(f"Loading training data from {args.data}")
    df, classes = preprocess(Path(args.data), Path(args.labels))
    packages = build_packages(df)
    if args.package not in packages:
        raise ValueError(f"Unknown package {args.package}. Choose from {PACKAGES}")
    features = packages[args.package]
    params = load_tuning_params(Path(args.tuning_dir), args.package)
    weights = np.array([params.pop(key) for key in WEIGHT_KEYS])

    if args.holdout:
        train_idx, _ = explore_commons.custom_train_test_split_2(
            classes, test_size=0.2, min_class_instances=40
        )
        X = features.iloc[train_idx]
        y_raw = classes.iloc[train_idx]
    else:
        X = features
        y_raw = classes

    y, mlb = explore_commons.classes_to_multilabel(
        [row.split(", ") for row in y_raw["class"]]
    )
    X_flat = flatten_features(X)

    model = xgb.XGBClassifier(
        **params,
        n_estimators=500,
        eta=0.1,
        tree_method="hist",
        random_state=SEED,
    )
    print(f"Fitting {args.package} model on {X_flat.shape[0]} stars, {X_flat.shape[1]} features")
    model.fit(X_flat, y, sample_weight=[np.sum(weights * row) for row in y])

    meta = {
        "package": args.package,
        "n_features": int(X_flat.shape[1]),
        "classes": list(mlb.classes_),
        "tuning_dir": str(args.tuning_dir),
        "n_estimators": 500,
        "eta": 0.1,
        "class_weights": weights.tolist(),
        "trained_on_all_labels": not args.holdout,
    }
    save_artifacts(Path(args.artifacts), model, mlb, features.columns, meta)
    print(f"Wrote artifacts to {args.artifacts}")
    print("Classes:", ", ".join(mlb.classes_))


def cmd_predict(args: argparse.Namespace) -> None:
    model, mlb, columns, meta = load_artifacts(Path(args.artifacts))
    labels_path = Path(args.labels) if args.labels else None
    df, _ = preprocess(Path(args.input), labels_path)
    packages = build_packages(df)
    package = args.package or meta.get("package", "Full")
    if package != meta.get("package"):
        print(
            f"Warning: scoring with package {package}, artifacts were trained as {meta.get('package')}"
        )
    features = packages[package].reindex(columns=columns)
    X = flatten_features(features)

    y_pred = np.asarray(model.predict(X))
    y_prob = probabilities_to_matrix(model.predict_proba(X), X.shape[0], len(mlb.classes_))
    y_pred = fill_empty_predictions(y_pred, y_prob)
    decoded = mlb.inverse_transform(y_pred)

    rows = []
    for star_id, labels, probs in zip(features.index, decoded, y_prob):
        rows.append(
            {
                "star_id": star_id,
                "predicted_classes": "; ".join(labels) if labels else "",
                **{f"prob_{cls}": float(p) for cls, p in zip(mlb.classes_, probs)},
            }
        )
    out = pd.DataFrame(rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Wrote {len(out)} predictions to {args.output}")
    else:
        print(out.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run the S-Phot stellar classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fit a model using saved tuning hyperparameters")
    train.add_argument("--package", default="Full", choices=PACKAGES)
    train.add_argument("--data", default=str(DEFAULT_TRAIN_DATA))
    train.add_argument("--labels", default=str(DEFAULT_TRAIN_LABELS))
    train.add_argument("--tuning-dir", default=str(DEFAULT_TUNING_DIR))
    train.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    train.add_argument(
        "--holdout",
        action="store_true",
        help="Train only on the 80% split from main.py instead of all labeled stars",
    )
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Score a new PySSED .dat file")
    predict.add_argument("--input", required=True, help="PySSED .dat (or .csv) with the same 9-line header")
    predict.add_argument("--labels", default=None, help="Optional labels file; dummy labels are used if omitted")
    predict.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    predict.add_argument("--package", default=None, choices=PACKAGES)
    predict.add_argument("--output", default=None, help="CSV path; prints a table if omitted")
    predict.set_defaults(func=cmd_predict)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
