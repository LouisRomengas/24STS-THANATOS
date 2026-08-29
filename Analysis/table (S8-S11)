import json
from pathlib import Path

import edsnlp
import numpy as np
import pandas as pd
from docx import Document

BASE = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results")
FINAL = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_datasets/FINAL")
PHYSICIAN = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_Resultats-annotations/train_val_entities/Tables_complications_characteristics.docx")
OUT = BASE / "Performances"

OUT.mkdir(parents=True, exist_ok=True)

CLASSIFIERS = {
    "svm": "Linear SVM",
    "logistic": "Logistic regression",
    "ordinal": "Ordinal logistic regression",
    "rf": "Random forest",
    "knn": "k-NN (cosine)",
}

SOURCES = [
    ("TF-IDF", "Classical", BASE / "tfidf_cd1to5", "classical", 0),
    ("Qwen3-Embedding-8B", "Embedding", BASE / "embeddings_qwen3e8b_cd1to5", "classical", 1),
    ("Gemma 3 4B", "Zero-shot", BASE / "llm_gemma3_4b_bf16_cd1to5", "llm", 3),
    ("MedGemma 4B", "Zero-shot", BASE / "llm_medgemma4b_bf16_cd1to5", "llm", 4),
    ("Ministral 3B", "Zero-shot", BASE / "llm_ministral3b_bf16_cd1to5", "llm", 5),
    ("Gemma 3 27B", "Zero-shot", BASE / "llm_gemma3_27b_bf16_cd1to5", "llm", 6),
    ("MedGemma 27B", "Zero-shot", BASE / "llm_medgemma27b_bf16_cd1to5", "llm", 7),
    ("Mistral Small 3.2 24B", "Zero-shot", BASE / "llm_mistral24b_bf16_cd1to5", "llm", 8),
    ("Gemma 3 4B", "QLoRA", BASE / "gemma3_4b_qlora_traineval_cd1to5", "qlora", 9),
    ("MedGemma 4B", "QLoRA", BASE / "medgemma_4b_qlora_traineval_cd1to5", "qlora", 10),
    ("Ministral 3B", "QLoRA", BASE / "lora_ministral3b_nf4_cd1to5", "qlora", 11),
    ("Gemma 3 27B", "QLoRA", BASE / "gemma3_27b_qlora_traineval_cd1to5", "qlora", 12),
    ("MedGemma 27B", "QLoRA", BASE / "medgemma27b_qlora_traineval_cd1to5", "qlora", 13),
    ("Mistral Small 3.2 24B", "QLoRA", BASE / "mistral24b_qlora_traineval_cd1to5", "qlora", 14),
]

validation = pd.read_csv(FINAL / "val_cd1to5.csv", low_memory=False)
validation["patient_id"] = validation["patient_id"].astype(str)
validation["cd_manuel"] = validation["cd_manuel"].astype(int)
validation["_report_id"] = np.arange(len(validation))

nlp = edsnlp.blank("eds")

for text, column in {
    "raw": "cr_text_brut",
    "cleaned": "cr_text_cleaned",
}.items():
    values = validation[column].fillna("").astype(str)
    counts = {value: len(nlp(value)) for value in values.drop_duplicates()}
    validation[f"n_tokens_{text}"] = values.map(counts).astype(int)

document = Document(PHYSICIAN)
table = document.tables[2]
physician_rows = []
patient_id = ""
n_complications = None

for row in table.rows[2:]:
    values = [
        " ".join(cell.text.replace("\xa0", " ").split())
        for cell in row.cells
    ]

    if len(values) >= 4:
        if values[0]:
            patient_id = values[0]

        if values[1]:
            value = pd.to_numeric(values[1].replace(",", "."), errors="coerce")

            if pd.notna(value):
                n_complications = int(value)

        if patient_id and n_complications is not None:
            physician_rows.append([patient_id, n_complications])

burden = (
    pd.DataFrame(
        physician_rows,
        columns=["patient_id", "n_complications"],
    )
    .drop_duplicates("patient_id", keep="first")
)

burden["patient_id"] = burden["patient_id"].astype(str)

validation = validation.merge(
    burden,
    on="patient_id",
    how="left",
)

validation["n_complications"] = validation["n_complications"].fillna(0).astype(int)

validation["complication_stratum"] = pd.cut(
    validation["n_complications"],
    bins=[-1, 0, 2, np.inf],
    labels=["0", "1–2", "≥3"],
)

cleaned_validation = validation.loc[validation["n_tokens_cleaned"] > 0].copy()

report_thresholds = {
    "raw": float(validation["n_tokens_raw"].median()),
    "cleaned": float(cleaned_validation["n_tokens_cleaned"].median()),
}

patient_raw = (
    validation.groupby("patient_id", as_index=False)
    .agg(
        n_tokens_raw=("n_tokens_raw", "sum"),
        n_complications=("n_complications", "first"),
    )
)

patient_cleaned = (
    cleaned_validation.groupby("patient_id", as_index=False)
    .agg(
        n_tokens_cleaned=("n_tokens_cleaned", "sum"),
        n_complications=("n_complications", "first"),
    )
)

patient_thresholds = {
    "raw": float(patient_raw["n_tokens_raw"].median()),
    "cleaned": float(patient_cleaned["n_tokens_cleaned"].median()),
}

candidates = []

for model, method, directory, schema, order in SOURCES:
    for path in sorted(directory.rglob("paper_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))

        if schema == "classical":
            text = "cleaned" if "cleaned" in summary["text_col"] else "raw"

            for classifier, block in summary["models"].items():
                if model == "TF-IDF":
                    approach = "tfidf"
                    approach_order = 0
                elif classifier == "knn":
                    approach = "qwen_knn"
                    approach_order = 2
                else:
                    approach = "qwen"
                    approach_order = 1

                candidates.append(
                    {
                        "approach": approach,
                        "approach_order": approach_order,
                        "Model": model,
                        "Method": CLASSIFIERS[classifier],
                        "Report text preprocessing": text,
                        "Prompting strategy": "N/A",
                        "patient_kappa": float(block["val_patient_level"]["kappa_quadratic"]),
                        "report_kappa": float(block["val_cr_level"]["kappa_quadratic"]),
                        "report_file": path.parent / f"val_predictions_cr_level_{classifier}.csv",
                    }
                )

        else:
            if schema == "llm":
                text = "cleaned" if "cleaned" in summary["text_col"] else "raw"
                strategy = summary["mode"]
            else:
                text = summary["text_mode"]
                strategy = summary["target_mode"]

            candidates.append(
                {
                    "approach": f"{model}_{method}",
                    "approach_order": order,
                    "Model": model,
                    "Method": method,
                    "Report text preprocessing": text,
                    "Prompting strategy": "One-step" if strategy == "direct" else "Two-step",
                    "patient_kappa": float(summary["val_results_patient_level"]["kappa_quadratic"]),
                    "report_kappa": float(summary["val_results_cr_level"]["kappa_quadratic"]),
                    "report_file": path.parent / "val_predictions_cr_level.csv",
                }
            )

candidates = pd.DataFrame(candidates)

patient_best = (
    candidates.sort_values(
        ["approach_order", "patient_kappa"],
        ascending=[True, False],
        kind="stable",
    )
    .drop_duplicates("approach", keep="first")
    .sort_values("approach_order", kind="stable")
)

report_best = (
    candidates.sort_values(
        ["approach_order", "report_kappa"],
        ascending=[True, False],
        kind="stable",
    )
    .drop_duplicates("approach", keep="first")
    .sort_values("approach_order", kind="stable")
)

patient_rows = []

for _, selected in patient_best.iterrows():
    predictions = pd.read_csv(selected["report_file"])
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["cd_manuel"] = predictions["cd_manuel"].astype(int)
    predictions["cd_pred"] = predictions["cd_pred"].astype(int)

    text = selected["Report text preprocessing"]

    if text == "cleaned" and len(predictions) == len(cleaned_validation):
        predictions["_report_id"] = cleaned_validation["_report_id"].to_numpy()
    else:
        predictions["_report_id"] = validation["_report_id"].to_numpy()

    reports = predictions.merge(
        validation[
            [
                "_report_id",
                "patient_id",
                "n_tokens_raw",
                "n_tokens_cleaned",
                "n_complications",
            ]
        ],
        on="_report_id",
        how="inner",
        suffixes=("", "_validation"),
    )

    if text == "cleaned":
        reports = reports.loc[reports["n_tokens_cleaned"] > 0].copy()

    patients = (
        reports.groupby("patient_id", as_index=False)
        .agg(
            cd_manuel=("cd_manuel", "max"),
            cd_pred=("cd_pred", "max"),
            n_tokens=(f"n_tokens_{text}", "sum"),
            n_complications=("n_complications", "first"),
        )
    )

    patients["absolute_error"] = (
        patients["cd_pred"] - patients["cd_manuel"]
    ).abs()

    patients["complication_stratum"] = pd.cut(
        patients["n_complications"],
        bins=[-1, 0, 2, np.inf],
        labels=["0", "1–2", "≥3"],
    )

    patients["length_stratum"] = np.where(
        patients["n_tokens"] <= patient_thresholds[text],
        "Short",
        "Long",
    )

    row = {
        "Model": selected["Model"],
        "Method": selected["Method"],
        "Report text preprocessing": text.title(),
        "Prompting strategy": selected["Prompting strategy"],
        "Overall": round(float(patients["absolute_error"].mean()), 2),
    }

    for stratum in ["0", "1–2", "≥3"]:
        row[f"Complication burden {stratum}"] = round(
            float(
                patients.loc[
                    patients["complication_stratum"] == stratum,
                    "absolute_error",
                ].mean()
            ),
            2,
        )

    for stratum in ["Short", "Long"]:
        row[f"Report length {stratum}"] = round(
            float(
                patients.loc[
                    patients["length_stratum"] == stratum,
                    "absolute_error",
                ].mean()
            ),
            2,
        )

    patient_rows.append(row)

report_rows = []

for _, selected in report_best.iterrows():
    predictions = pd.read_csv(selected["report_file"])
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["cd_manuel"] = predictions["cd_manuel"].astype(int)
    predictions["cd_pred"] = predictions["cd_pred"].astype(int)

    text = selected["Report text preprocessing"]

    if text == "cleaned" and len(predictions) == len(cleaned_validation):
        predictions["_report_id"] = cleaned_validation["_report_id"].to_numpy()
    else:
        predictions["_report_id"] = validation["_report_id"].to_numpy()

    reports = predictions.merge(
        validation[
            [
                "_report_id",
                "n_tokens_raw",
                "n_tokens_cleaned",
                "n_complications",
            ]
        ],
        on="_report_id",
        how="inner",
    )

    if text == "cleaned":
        reports = reports.loc[reports["n_tokens_cleaned"] > 0].copy()

    reports["absolute_error"] = (
        reports["cd_pred"] - reports["cd_manuel"]
    ).abs()

    reports["complication_stratum"] = pd.cut(
        reports["n_complications"],
        bins=[-1, 0, 2, np.inf],
        labels=["0", "1–2", "≥3"],
    )

    reports["length_stratum"] = np.where(
        reports[f"n_tokens_{text}"] <= report_thresholds[text],
        "Short",
        "Long",
    )

    row = {
        "Model": selected["Model"],
        "Method": selected["Method"],
        "Report text preprocessing": text.title(),
        "Prompting strategy": selected["Prompting strategy"],
        "Overall": round(float(reports["absolute_error"].mean()), 2),
    }

    for stratum in ["0", "1–2", "≥3"]:
        row[f"Complication burden {stratum}"] = round(
            float(
                reports.loc[
                    reports["complication_stratum"] == stratum,
                    "absolute_error",
                ].mean()
            ),
            2,
        )

    for stratum in ["Short", "Long"]:
        row[f"Report length {stratum}"] = round(
            float(
                reports.loc[
                    reports["length_stratum"] == stratum,
                    "absolute_error",
                ].mean()
            ),
            2,
        )

    report_rows.append(row)

pd.DataFrame(patient_rows).to_csv(
    OUT / "Table-S8-Errors-Subgroups-patient-level.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(report_rows).to_csv(
    OUT / "Table-S11-Errors-Subgroups-report-level.csv",
    index=False,
    encoding="utf-8-sig",
)
