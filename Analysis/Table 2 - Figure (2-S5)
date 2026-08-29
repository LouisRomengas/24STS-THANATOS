import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results")
OUT = BASE / "Performances"

OUT.mkdir(parents=True, exist_ok=True)

CLASSIFIER_LABELS = {
    "logistic": "logistic regression",
    "svm_linear": "linear SVM",
    "svm": "linear SVM",
    "random_forest": "random forest",
    "rf": "random forest",
    "ordinal": "ordinal logistic regression",
    "knn_cosine": "k-NN (cosine)",
}

MODEL_ORDER = [
    "Gemma 3 4B",
    "MedGemma 4B",
    "Ministral 3B",
    "Gemma 3 27B",
    "MedGemma 27B",
    "Mistral Small 3.2 24B",
]

COLORS = {
    ("Zero-shot", "direct"): "#9DC3E6",
    ("Zero-shot", "twostep"): "#2E5A88",
    ("QLoRA", "direct"): "#8ED9D0",
    ("QLoRA", "twostep"): "#207A74",
}

HATCHES = {"raw": None, "cleaned": "////"}

SOURCES = [
    ("tfidf", "TF-IDF", None, BASE / "tfidf_robust_cd1to5", "classical", 0),
    ("qwen", "Qwen3-Embedding-8B", None, BASE / "embeddings_qwen3e8b_cd1to5", "classical", 1),
    ("qwen_knn", "Qwen3-Embedding-8B (k-NN)", None, BASE / "embeddings_qwen3e8b_knn_cd1to5", "classical", 2),
    ("gemma4_zero", "Gemma 3 4B", "Zero-shot", BASE / "llm_gemma3_4b_cd1to5_structured", "llm", 3),
    ("gemma4_qlora", "Gemma 3 4B", "QLoRA", BASE / "gemma3_4b_qlora_traineval_cd1to5", "llm", 4),
    ("medgemma4_zero", "MedGemma 4B", "Zero-shot", BASE / "llm_medgemma4b_cd1to5_structured", "llm", 5),
    ("medgemma4_qlora", "MedGemma 4B", "QLoRA", BASE / "medgemma_4b_qlora_traineval_cd1to5", "llm", 6),
    ("ministral_zero", "Ministral 3B", "Zero-shot", BASE / "llm_ministral3b_bf16_robust_cd1to5_structured_vllm", "llm", 7),
    ("ministral_qlora", "Ministral 3B", "QLoRA", BASE / "lora_ministral3b_nf4_cd1to5", "llm", 8),
    ("gemma27_zero", "Gemma 3 27B", "Zero-shot", BASE / "llm_gemma3_27b_cd1to5_structured", "llm", 9),
    ("gemma27_qlora", "Gemma 3 27B", "QLoRA", BASE / "gemma3_27b_qlora_traineval_cd1to5", "llm", 10),
    ("medgemma27_zero", "MedGemma 27B", "Zero-shot", BASE / "llm_medgemma27b_cd1to5_structured", "llm", 11),
    ("medgemma27_qlora", "MedGemma 27B", "QLoRA", BASE / "medgemma27b_qlora_traineval_cd1to5", "llm", 12),
    ("mistral_zero", "Mistral Small 3.2 24B", "Zero-shot", BASE / "llm_mistral24b_cd1to5_structured_vllm", "llm", 13),
    ("mistral_qlora", "Mistral Small 3.2 24B", "QLoRA", BASE / "mistral24b_qlora_traineval_cd1to5", "llm", 14),
]

METRICS = [
    "kappa_quadratic",
    "balanced_accuracy",
    "macro_f1",
    "severe_sensitivity",
]

FIGURE_METRICS = [
    "kappa_quadratic",
    "balanced_accuracy",
    "macro_f1",
]

FIGURE_LABELS = {
    "kappa_quadratic": "Quadratic weighted κ",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
}

calibration = json.loads(
    (OUT / "calibration_summary.json").read_text(encoding="utf-8")
)

reference = {}

for metric in METRICS:
    bootstrap = calibration.get(f"bootstrap_{metric}")

    if metric in calibration and bootstrap:
        reference[metric] = {
            "value": float(calibration[metric]),
            "low": float(bootstrap["ci95_low"]),
            "high": float(bootstrap["ci95_high"]),
        }

rows = []

for source, model, paradigm, directory, schema, order in SOURCES:
    for path in sorted(directory.rglob("paper_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        text = (
            summary.get("text_mode")
            or summary.get("text_tag")
            or (
                "cleaned"
                if "cleaned" in summary.get("text_col", "")
                else "raw"
            )
        )
        strategy = summary.get("target_mode") or summary.get("mode") or ""

        if schema == "classical":
            results = summary.get("all_classifier_results") or summary.get("models") or {}

            for classifier, block in results.items():
                row = {
                    "source": source,
                    "model": model,
                    "paradigm": "Classical",
                    "classifier": classifier,
                    "text": text,
                    "strategy": "",
                    "order": order,
                }

                for level, point_keys, ci_keys in [
                    (
                        "patient",
                        ["val_patient_level"],
                        ["bootstrap_patient_level", "bootstrap_val_patient_level"],
                    ),
                    (
                        "report",
                        ["val_cr_level"],
                        ["bootstrap_cr_level", "bootstrap_val_cr_level"],
                    ),
                ]:
                    point = next(
                        (
                            block[key]
                            for key in point_keys
                            if key in block
                        ),
                        {},
                    )
                    ci = next(
                        (
                            block[key]
                            for key in ci_keys
                            if key in block
                        ),
                        {},
                    )

                    for metric in METRICS:
                        row[f"{level}_{metric}"] = point.get(metric, np.nan)
                        metric_ci = ci.get(metric) or {}
                        row[f"{level}_{metric}_low"] = metric_ci.get("ci95_low", np.nan)
                        row[f"{level}_{metric}_high"] = metric_ci.get("ci95_high", np.nan)

                rows.append(row)

        else:
            row = {
                "source": source,
                "model": model,
                "paradigm": paradigm,
                "classifier": "",
                "text": text,
                "strategy": strategy,
                "order": order,
            }

            for level, point_key, ci_key in [
                ("patient", "val_results_patient_level", "bootstrap_val_patient_level"),
                ("report", "val_results_cr_level", "bootstrap_val_cr_level"),
            ]:
                point = summary.get(point_key) or {}
                ci = summary.get(ci_key) or {}

                for metric in METRICS:
                    row[f"{level}_{metric}"] = point.get(metric, np.nan)
                    metric_ci = ci.get(metric) or {}
                    row[f"{level}_{metric}_low"] = metric_ci.get("ci95_low", np.nan)
                    row[f"{level}_{metric}_high"] = metric_ci.get("ci95_high", np.nan)

            rows.append(row)

performance = pd.DataFrame(rows)

best = (
    performance.loc[
        performance["model"] != "Qwen3-Embedding-8B (k-NN)"
    ]
    .sort_values(
        ["order", "patient_kappa_quadratic"],
        ascending=[True, False],
        kind="stable",
    )
    .drop_duplicates("source", keep="first")
    .sort_values("order", kind="stable")
    .copy()
)

best["classifier"] = best["classifier"].map(CLASSIFIER_LABELS).fillna(
    best["classifier"]
)

calibration_row = {
    "model": "Inter-annotator agreement",
    "paradigm": "Reference",
    "classifier": "",
    "text": "",
    "strategy": "",
}

for metric in METRICS:
    calibration_row[f"patient_{metric}"] = (
        reference[metric]["value"] if metric in reference else np.nan
    )
    calibration_row[f"patient_{metric}_low"] = (
        reference[metric]["low"] if metric in reference else np.nan
    )
    calibration_row[f"patient_{metric}_high"] = (
        reference[metric]["high"] if metric in reference else np.nan
    )

table2 = pd.concat(
    [
        pd.DataFrame([calibration_row]),
        best[
            [
                "model",
                "paradigm",
                "classifier",
                "text",
                "strategy",
                *[
                    column
                    for metric in METRICS
                    for column in [
                        f"patient_{metric}",
                        f"patient_{metric}_low",
                        f"patient_{metric}_high",
                    ]
                ],
            ]
        ],
    ],
    ignore_index=True,
)

table2.to_csv(
    OUT / "Table2-data.csv",
    index=False,
    encoding="utf-8-sig",
)

plt.rcParams["hatch.linewidth"] = 1.2

for level, title, filename in [
    ("report", "Report level", "Figure S5 - model comparison report-level"),
    ("patient", "Patient level", "Figure 2 - model comparison patient-level"),
]:
    llm = performance.loc[
        performance["paradigm"].isin(["Zero-shot", "QLoRA"])
    ]

    present = [
        model
        for model in MODEL_ORDER
        if (llm["model"] == model).any()
    ]

    classical = {}

    for model in ["TF-IDF", "Qwen3-Embedding-8B"]:
        candidates = performance.loc[
            (performance["model"] == model)
            & performance[f"{level}_kappa_quadratic"].notna()
        ].sort_values(
            f"{level}_kappa_quadratic",
            ascending=False,
            kind="stable",
        )

        if len(candidates):
            classical[model] = candidates.iloc[0]

    ordered = {}

    for model in present:
        ordered[model] = (
            llm.loc[
                (llm["model"] == model)
                & llm[f"{level}_kappa_quadratic"].notna()
            ]
            .sort_values(
                f"{level}_kappa_quadratic",
                ascending=True,
                kind="stable",
            )
        )

    max_bars = max(len(entries) for entries in ordered.values())
    x = np.arange(len(present))
    bar_width = 0.80 / max_bars

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(max(11.0, 2.2 * len(present)), 11.0),
        sharex=True,
    )

    for axis, metric in zip(axes, FIGURE_METRICS):
        if metric in reference:
            axis.axhspan(
                reference[metric]["low"],
                reference[metric]["high"],
                color="#E8820C",
                alpha=0.12,
                zorder=0,
            )
            axis.axhline(
                reference[metric]["value"],
                color="#E8820C",
                linestyle="-",
                linewidth=1.4,
                zorder=1,
            )

        for position, model in enumerate(present):
            entries = ordered[model]
            offsets = (
                np.arange(len(entries))
                - (len(entries) - 1) / 2
            ) * bar_width

            for index, (_, row) in enumerate(entries.iterrows()):
                value = row[f"{level}_{metric}"]
                low = row[f"{level}_{metric}_low"]
                high = row[f"{level}_{metric}_high"]

                axis.bar(
                    x[position] + offsets[index],
                    value,
                    bar_width,
                    color=COLORS[(row["paradigm"], row["strategy"])],
                    hatch=HATCHES[row["text"]],
                    edgecolor="white",
                    linewidth=0.5,
                    yerr=np.array(
                        [
                            [max(0, value - low)],
                            [max(0, high - value)],
                        ]
                    ),
                    capsize=1.5,
                    error_kw={
                        "elinewidth": 0.7,
                        "ecolor": "#333333",
                    },
                    zorder=2,
                )

        for model, color, style in [
            ("TF-IDF", "#8C8C8C", "--"),
            ("Qwen3-Embedding-8B", "#4C9F70", ":"),
        ]:
            if model in classical:
                axis.axhline(
                    classical[model][f"{level}_{metric}"],
                    color=color,
                    linestyle=style,
                    linewidth=1.3,
                    zorder=3,
                )

        for boundary in np.arange(len(present) - 1) + 0.5:
            axis.axvline(
                boundary,
                color="#DDDDDD",
                linewidth=0.8,
                zorder=0,
            )

        axis.set_ylabel(FIGURE_LABELS[metric], fontsize=10)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(present, fontsize=10)
    axes[-1].set_xlim(-0.6, len(present) - 0.4)

    handles = [
        Patch(
            facecolor=COLORS[("Zero-shot", "direct")],
            label="Zero-shot, one-step",
        ),
        Patch(
            facecolor=COLORS[("Zero-shot", "twostep")],
            label="Zero-shot, two-step",
        ),
        Patch(
            facecolor=COLORS[("QLoRA", "direct")],
            label="QLoRA, one-step",
        ),
        Patch(
            facecolor=COLORS[("QLoRA", "twostep")],
            label="QLoRA, two-step",
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="#404040",
            linewidth=0.4,
            label="Raw text",
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="#404040",
            linewidth=0.4,
            hatch="////",
            label="Cleaned text",
        ),
    ]

    for model, color, style in [
        ("TF-IDF", "#8C8C8C", "--"),
        ("Qwen3-Embedding-8B", "#4C9F70", ":"),
    ]:
        if model in classical:
            row = classical[model]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=style,
                    label=(
                        f"{model}, "
                        f"{CLASSIFIER_LABELS.get(row['classifier'], row['classifier'])} "
                        f"({row['text']})"
                    ),
                )
            )

    if reference:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#E8820C",
                linestyle="-",
                label="Inter-annotator agreement (metric-specific)",
            )
        )

    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(
        OUT / f"{filename}.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        OUT / f"{filename}.pdf",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)
