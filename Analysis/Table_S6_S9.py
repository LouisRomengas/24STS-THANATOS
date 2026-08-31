import json
from pathlib import Path

import pandas as pd

BASE = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_results"
)
OUT = BASE / "Performances"
OUT.mkdir(parents=True, exist_ok=True)

CLASSIFIERS = {
    "svm": ("Linear SVM", 0),
    "logistic": ("Logistic regression", 1),
    "ordinal": ("Ordinal logistic regression", 2),
    "rf": ("Random forest", 3),
    "knn": ("k-NN (cosine)", 4),
}

SOURCES = [
    ("TF-IDF", "Classical", BASE / "tfidf_cd1to5", "classical", 0),
    ("Qwen3-Embedding-8B", "Embedding", BASE / "embeddings_qwen3e8b_cd1to5", "classical", 1),
    ("Gemma 3 4B", "General LLM", BASE / "llm_gemma3_4b_bf16_cd1to5", "llm", 2),
    ("MedGemma 4B", "Medical LLM", BASE / "llm_medgemma4b_bf16_cd1to5", "llm", 3),
    ("Ministral 3B", "General LLM", BASE / "llm_ministral3b_bf16_cd1to5", "llm", 4),
    ("Gemma 3 27B", "General LLM", BASE / "llm_gemma3_27b_bf16_cd1to5", "llm", 5),
    ("MedGemma 27B", "Medical LLM", BASE / "llm_medgemma27b_bf16_cd1to5", "llm", 6),
    ("Mistral Small 3.2 24B", "General LLM", BASE / "llm_mistral24b_bf16_cd1to5", "llm", 7),
    ("Gemma 3 4B (QLoRA)", "General LLM", BASE / "gemma3_4b_qlora_traineval_cd1to5", "qlora", 8),
    ("MedGemma 4B (QLoRA)", "Medical LLM", BASE / "medgemma_4b_qlora_traineval_cd1to5", "qlora", 9),
    ("Ministral 3B (QLoRA)", "General LLM", BASE / "lora_ministral3b_nf4_cd1to5", "qlora", 10),
    ("Gemma 3 27B (QLoRA)", "General LLM", BASE / "gemma3_27b_qlora_traineval_cd1to5", "qlora", 11),
    ("MedGemma 27B (QLoRA)", "Medical LLM", BASE / "medgemma27b_qlora_traineval_cd1to5", "qlora", 12),
    ("Mistral Small 3.2 24B (QLoRA)", "General LLM", BASE / "mistral24b_qlora_traineval_cd1to5", "qlora", 13),
]

METRICS = {
    "kappa_quadratic": "Quadratic weighted kappa [95% CI]",
    "balanced_accuracy": "Balanced accuracy [95% CI]",
    "macro_f1": "Macro-F1 [95% CI]",
    "severe_sensitivity": "Sensitivity, CD grade≥III [95% CI]",
}

patient_rows = []
report_rows = []

for model, family, directory, schema, model_order in SOURCES:
    summaries = sorted(directory.rglob("paper_summary.json"))

    if schema == "classical":

        for path in summaries:
            summary = json.loads(path.read_text(encoding="utf-8"))
            text_mode = "cleaned" if "cleaned" in summary["text_col"] else "raw"
            models = summary["models"]
            expected = {"svm", "logistic", "ordinal", "rf"}

            if model == "Qwen3-Embedding-8B":
                expected = expected | {"knn"}

            for classifier in ["svm", "logistic", "ordinal", "rf", "knn"]:
                if classifier not in models:
                    continue

                base_row = {
                    "_model_order": model_order,
                    "_classifier_order": CLASSIFIERS[classifier][1],
                    "_text_order": 0 if text_mode == "cleaned" else 1,
                    "_strategy_order": 0,
                    "Model": model,
                    "Family": family,
                    "Classifier": CLASSIFIERS[classifier][0],
                    "Report text preprocessing": text_mode,
                    "Prompting strategy": "N/A",
                }

                patient = base_row.copy()
                report = base_row.copy()

                patient_results = models[classifier]["val_patient_level"]
                patient_bootstrap = models[classifier]["bootstrap_val_patient_level"]
                report_results = models[classifier]["val_cr_level"]
                report_bootstrap = models[classifier]["bootstrap_val_cr_level"]

                for metric, column in METRICS.items():
                    patient[column] = (
                        f'{float(patient_results[metric]):.3f} '
                        f'[{float(patient_bootstrap[metric]["ci95_low"]):.3f}; '
                        f'{float(patient_bootstrap[metric]["ci95_high"]):.3f}]'
                    )
                    report[column] = (
                        f'{float(report_results[metric]):.3f} '
                        f'[{float(report_bootstrap[metric]["ci95_low"]):.3f}; '
                        f'{float(report_bootstrap[metric]["ci95_high"]):.3f}]'
                    )

                patient_rows.append(patient)
                report_rows.append(report)

    else:
        seen = set()

        for path in summaries:
            summary = json.loads(path.read_text(encoding="utf-8"))

            if schema == "llm":
                text_mode = "cleaned" if "cleaned" in summary["text_col"] else "raw"
                target_mode = summary["mode"]
            else:
                text_mode = summary["text_mode"]
                target_mode = summary["target_mode"]

            seen.add((text_mode, target_mode))

            base_row = {
                "_model_order": model_order,
                "_classifier_order": 0,
                "_text_order": 0 if text_mode == "cleaned" else 1,
                "_strategy_order": 0 if target_mode == "direct" else 1,
                "Model": model,
                "Family": family,
                "Classifier": "N/A",
                "Report text preprocessing": text_mode,
                "Prompting strategy": "One-step" if target_mode == "direct" else "Two-step",
            }

            patient = base_row.copy()
            report = base_row.copy()

            patient_results = summary["val_results_patient_level"]
            patient_bootstrap = summary["bootstrap_val_patient_level"]
            report_results = summary["val_results_cr_level"]
            report_bootstrap = summary["bootstrap_val_cr_level"]

            for metric, column in METRICS.items():
                patient[column] = (
                    f'{float(patient_results[metric]):.3f} '
                    f'[{float(patient_bootstrap[metric]["ci95_low"]):.3f}; '
                    f'{float(patient_bootstrap[metric]["ci95_high"]):.3f}]'
                )
                report[column] = (
                    f'{float(report_results[metric]):.3f} '
                    f'[{float(report_bootstrap[metric]["ci95_low"]):.3f}; '
                    f'{float(report_bootstrap[metric]["ci95_high"]):.3f}]'
                )

            patient_rows.append(patient)
            report_rows.append(report)
        }

sort_columns = [
    "_model_order",
    "_classifier_order",
    "_text_order",
    "_strategy_order",
]

drop_columns = sort_columns

table_s6 = (
    pd.DataFrame(patient_rows)
    .sort_values(sort_columns, kind="stable")
    .drop(columns=drop_columns)
    .reset_index(drop=True)
)

table_s9 = (
    pd.DataFrame(report_rows)
    .sort_values(sort_columns, kind="stable")
    .drop(columns=drop_columns)
    .reset_index(drop=True)
)

columns = [
    "Model",
    "Family",
    "Classifier",
    "Report text preprocessing",
    "Prompting strategy",
    "Quadratic weighted kappa [95% CI]",
    "Balanced accuracy [95% CI]",
    "Macro-F1 [95% CI]",
    "Sensitivity, CD grade≥III [95% CI]",
]

table_s6.to_csv(
    OUT / "Table-S6-Performances-patient-level.csv",
    index=False,
    encoding="utf-8-sig",
)

table_s9.to_csv(
    OUT / "Table-S9-Performances-report-level.csv",
    index=False,
    encoding="utf-8-sig",
)
