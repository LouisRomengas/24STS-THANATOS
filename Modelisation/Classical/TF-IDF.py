import argparse
import json
import time
from pathlib import Path

import mord
import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/FINAL"
)

TRAIN_PATH = DATA_DIR / "train_cd1to5.csv"
VAL_PATH = DATA_DIR / "val_cd1to5.csv"

OUTPUT_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_results/tfidf_cd1to5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [1, 2, 3, 4, 5]
LABEL_NAMES = ["CD-I", "CD-II", "CD-III", "CD-IV", "CD-V"]

TFIDF_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4)],
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__min_df": [1, 2],
    "tfidf__max_df": [0.90, 0.95],
}

MODELS = {
    "logistic": LogisticRegression(
        max_iter=20000,
        random_state=42,
        class_weight="balanced",
    ),
    "svm": LinearSVC(
        max_iter=20000,
        random_state=42,
        dual=False,
        class_weight="balanced",
    ),
    "rf": RandomForestClassifier(
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    ),
    "ordinal": mord.LogisticAT(
        alpha=1.0,
        max_iter=1000,
    ),
}


def scores(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    severe_true = y_true >= 3
    severe_pred = y_pred >= 3
    denominator = int(severe_true.sum())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "severe_sensitivity": float(
            (severe_true & severe_pred).sum() / denominator
        ) if denominator else 0.0,
    }


def bootstrap(y_true, y_pred, n_bootstrap=1000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in scores(y_true, y_pred)}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        result = scores(y_true[idx], y_pred[idx])

        for key in values:
            values[key].append(result[key])

    return {
        key: {
            "mean": float(np.nanmean(value)),
            "std": float(np.nanstd(value, ddof=1)),
            "ci95_low": float(np.nanpercentile(value, 2.5)),
            "ci95_high": float(np.nanpercentile(value, 97.5)),
            "n_bootstrap": n_bootstrap,
        }
        for key, value in values.items()
    }


def run(text_col, tag):
    train = pd.read_csv(TRAIN_PATH, usecols=["patient_id", text_col, "cd_manuel"])
    val = pd.read_csv(VAL_PATH, usecols=["patient_id", text_col, "cd_manuel"])

    for df in (train, val):
        df[text_col] = df[text_col].fillna("").astype(str)
        df["patient_id"] = df["patient_id"].fillna("").astype(str).str.strip()
        df["cd_manuel"] = pd.to_numeric(df["cd_manuel"], errors="coerce")
        df.dropna(subset=["cd_manuel"], inplace=True)
        df.drop(df[~df["cd_manuel"].between(1, 5)].index, inplace=True)
        df.drop(df[df[text_col].str.strip().eq("")].index, inplace=True)
        df["cd_manuel"] = df["cd_manuel"].astype(int)

    assert not set(train["patient_id"]) & set(val["patient_id"])

    X_train = train[text_col].to_numpy(dtype=object)
    y_train = train["cd_manuel"].to_numpy(dtype=int)
    groups = train["patient_id"].to_numpy(dtype=object)
    X_val = val[text_col].to_numpy(dtype=object)
    y_val = val["cd_manuel"].to_numpy(dtype=int)

    result_dir = OUTPUT_DIR / f"text_{tag}"
    result_dir.mkdir(parents=True, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        project_name="THANATOS",
        experiment_id=f"tfidf_{tag}",
        output_dir=str(result_dir),
        output_file="emissions.csv",
        log_level="critical",
        tracking_mode="process",
        country_iso_code="FRA",
    )

    tracker.start()
    start = time.perf_counter()

    baseline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10000,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=20000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    stability = {key: [] for key in scores(y_train, y_train)}

    for repeat in range(10):
        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=1000 + repeat,
        )

        for train_idx, test_idx in cv.split(X_train, y_train, groups):
            baseline.fit(X_train[train_idx], y_train[train_idx])
            pred = baseline.predict(X_train[test_idx])
            fold = scores(y_train[test_idx], pred)

            for key in stability:
                stability[key].append(fold[key])

    stability_summary = {
        key: {
            "mean": float(np.mean(value)),
            "std": float(np.std(value, ddof=1)),
            "n_scores": len(value),
        }
        for key, value in stability.items()
    }

    scorer = make_scorer(
        cohen_kappa_score,
        weights="quadratic",
    )

    model_results = {}
    best_name = None
    best_score = -np.inf

    for name, classifier in MODELS.items():
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )

        grid = GridSearchCV(
            pipeline,
            TFIDF_GRID,
            scoring=scorer,
            cv=StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            ),
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        fit_kwargs = {}

        if name == "ordinal":
            fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

        grid.fit(
            X_train,
            y_train,
            groups=groups,
            **fit_kwargs,
        )

        pd.DataFrame(grid.cv_results_).to_csv(
            result_dir / f"grid_results_{name}.csv",
            index=False,
        )

        pred = grid.best_estimator_.predict(X_val)
        cr = scores(y_val, pred)
        cr["confusion_matrix"] = confusion_matrix(
            y_val,
            pred,
            labels=LABELS,
        ).tolist()
        cr["classification_report"] = classification_report(
            y_val,
            pred,
            labels=LABELS,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )

        report = val.copy()
        report["cd_pred"] = pred
        report.to_csv(
            result_dir / f"val_predictions_cr_level_{name}.csv",
            index=False,
        )

        patient = (
            report.groupby("patient_id", as_index=False)
            .agg(
            cd_manuel=("cd_manuel", "max"),
            cd_pred=("cd_pred", "max"),
        )
        )

        patient.to_csv(
            result_dir / f"val_predictions_patient_level_{name}.csv",
            index=False,
        )

        patient_metrics = scores(
            patient["cd_manuel"],
            patient["cd_pred"],
        )
        patient_metrics["confusion_matrix"] = confusion_matrix(
            patient["cd_manuel"],
            patient["cd_pred"],
            labels=LABELS,
        ).tolist()
        patient_metrics["classification_report"] = classification_report(
            patient["cd_manuel"],
            patient["cd_pred"],
            labels=LABELS,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )

        model_results[name] = {
            "best_params": grid.best_params_,
            "best_cv_kappa_quadratic": float(grid.best_score_),
            "val_cr_level": cr,
            "val_patient_level": patient_metrics,
            "bootstrap_val_cr_level": bootstrap(y_val, pred),
            "bootstrap_val_patient_level": bootstrap(
                patient["cd_manuel"],
                patient["cd_pred"],
            ),
        }

        if grid.best_score_ > best_score:
            best_score = float(grid.best_score_)
            best_name = name

    tracker.stop()
    wall_seconds = time.perf_counter() - start

    emissions = {}
    emissions_path = result_dir / "emissions.csv"

    if emissions_path.exists():
        energy = pd.read_csv(emissions_path)

        if not energy.empty:
            row = energy.iloc[-1]
            emissions = {
                "duration_seconds": float(row.get("duration", 0)),
                "energy_kwh": float(row.get("energy_consumed", 0)),
                "emissions_kg_co2eq": float(row.get("emissions", 0)),
            }

    summary = {
        "train_csv": str(TRAIN_PATH),
        "external_test_csv": str(VAL_PATH),
        "text_col": text_col,
        "clinical_context": "text_only",
        "tfidf_grid": {
            "ngram_range": [[1, 1], [1, 2], [1, 3], [1, 4]],
            "max_features": [5000, 10000, 20000],
            "min_df": [1, 2],
            "max_df": [0.90, 0.95],
            "sublinear_tf": True,
        },
        "classifiers": ["logistic", "svm", "rf", "ordinal"],
        "class_weighting": "inverse_class_frequency",
        "model_selection": {
            "cv": "StratifiedGroupKFold",
            "n_splits": 5,
            "group": "patient_id",
            "criterion": "quadratic_weighted_kappa",
            "best_classifier": best_name,
            "best_cv_kappa_quadratic": best_score,
        },
        "stability_assessment": {
            "model": "logistic_regression",
            "folds": 5,
            "repeats": 10,
            "configuration": {
                "ngram_range": [1, 2],
                "max_features": 10000,
                "min_df": 2,
                "max_df": 0.95,
                "sublinear_tf": True,
            },
            "results": stability_summary,
        },
        "models": model_results,
        "wall_seconds": wall_seconds,
        "energy": emissions,
    }

    (result_dir / "paper_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--text_cols",
    nargs="+",
    default=["cleaned", "raw"],
    choices=["cleaned", "raw"],
)
args = parser.parse_args()

columns = {
    "cleaned": "cr_text_cleaned",
    "raw": "cr_text_brut",
}

for text_key in args.text_cols:
    run(columns[text_key], text_key)
