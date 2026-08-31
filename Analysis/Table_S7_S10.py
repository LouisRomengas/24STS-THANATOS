import json
from pathlib import Path

import pandas as pd

BASE = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results")
FINAL = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_datasets/FINAL")
OUT = BASE / "Performances"

OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    1: "CD-I",
    2: "CD-II",
    3: "CD-III",
    4: "CD-IV",
    5: "CD-V",
}

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
patient_reference = (
    validation.groupby("patient_id", as_index=False)
    .agg(cd_manuel=("cd_manuel", "max"))
)

report_support = validation["cd_manuel"].value_counts().reindex(LABELS).fillna(0).astype(int)
patient_support = patient_reference["cd_manuel"].value_counts().reindex(LABELS).fillna(0).astype(int)

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
                        "model_order": order,
                        "Model": model,
                        "Method": CLASSIFIERS[classifier],
                        "Report text preprocessing": text,
                        "Prompting strategy": "N/A",
                        "patient_kappa": float(block["val_patient_level"]["kappa_quadratic"]),
                        "report_kappa": float(block["val_cr_level"]["kappa_quadratic"]),
                        "patient_file": path.parent / f"val_predictions_patient_level_{classifier}.csv",
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

            approach = f"{model}_{method}"

            candidates.append(
                {
                    "approach": approach,
                    "approach_order": order,
                    "model_order": order,
                    "Model": model,
                    "Method": method,
                    "Report text preprocessing": text,
                    "Prompting strategy": "One-step" if strategy == "direct" else "Two-step",
                    "patient_kappa": float(summary["val_results_patient_level"]["kappa_quadratic"]),
                    "report_kappa": float(summary["val_results_cr_level"]["kappa_quadratic"]),
                    "patient_file": path.parent / "val_predictions_patient_level.csv",
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
    predictions = pd.read_csv(selected["patient_file"])
    row = {
        "Model": selected["Model"],
        "Method": selected["Method"],
        "Report text preprocessing": selected["Report text preprocessing"].title(),
        "Prompting strategy": selected["Prompting strategy"],
    }

    for grade, label in LABELS.items():
        block = predictions.loc[predictions["cd_manuel"] == grade]
        row[f"{label} (n={patient_support[grade]})"] = round(
            float((block["cd_pred"] == grade).mean()),
            2,
        )

    patient_rows.append(row)

report_rows = []

for _, selected in report_best.iterrows():
    predictions = pd.read_csv(selected["report_file"])
    row = {
        "Model": selected["Model"],
        "Method": selected["Method"],
        "Report text preprocessing": selected["Report text preprocessing"].title(),
        "Prompting strategy": selected["Prompting strategy"],
    }

    if selected["Model"] == "TF-IDF" and selected["Method"] == "Logistic regression":
        row["Method"] = "LR"

    for grade, label in LABELS.items():
        block = predictions.loc[predictions["cd_manuel"] == grade]
        row[f"{label} (n={report_support[grade]})"] = round(
            float((block["cd_pred"] == grade).mean()),
            2,
        )

    report_rows.append(row)

pd.DataFrame(patient_rows).to_csv(
    OUT / "Table-S7-Recall-patient-level.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(report_rows).to_csv(
    OUT / "Table-S10-Recall-report-level.csv",
    index=False,
    encoding="utf-8-sig",
)
