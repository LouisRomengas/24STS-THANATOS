import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results")
FINAL = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_datasets/FINAL")
OUT = BASE / "Performances"
B = 1000
SEED = 42
LABELS = [1, 2, 3, 4, 5]

OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("Gemma 3 4B", "Gemma 3", "small", "Zero-shot", BASE / "llm_gemma3_4b_bf16_cd1to5", "llm"),
    ("MedGemma 4B", "MedGemma", "small", "Zero-shot", BASE / "llm_medgemma4b_bf16_cd1to5", "llm"),
    ("Ministral 3B", "Mistral", "small", "Zero-shot", BASE / "llm_ministral3b_bf16_cd1to5", "llm"),
    ("Gemma 3 27B", "Gemma 3", "large", "Zero-shot", BASE / "llm_gemma3_27b_bf16_cd1to5", "llm"),
    ("MedGemma 27B", "MedGemma", "large", "Zero-shot", BASE / "llm_medgemma27b_bf16_cd1to5", "llm"),
    ("Mistral Small 3.2 24B", "Mistral", "large", "Zero-shot", BASE / "llm_mistral24b_bf16_cd1to5", "llm"),
    ("Gemma 3 4B", "Gemma 3", "small", "QLoRA", BASE / "gemma3_4b_qlora_traineval_cd1to5", "qlora"),
    ("MedGemma 4B", "MedGemma", "small", "QLoRA", BASE / "medgemma_4b_qlora_traineval_cd1to5", "qlora"),
    ("Ministral 3B", "Mistral", "small", "QLoRA", BASE / "lora_ministral3b_nf4_cd1to5", "qlora"),
    ("Gemma 3 27B", "Gemma 3", "large", "QLoRA", BASE / "gemma3_27b_qlora_traineval_cd1to5", "qlora"),
    ("MedGemma 27B", "MedGemma", "large", "QLoRA", BASE / "medgemma27b_qlora_traineval_cd1to5", "qlora"),
    ("Mistral Small 3.2 24B", "Mistral", "large", "QLoRA", BASE / "mistral24b_qlora_traineval_cd1to5", "qlora"),
]

MODEL_ORDER = [
    "Gemma 3 4B",
    "MedGemma 4B",
    "Ministral 3B",
    "Gemma 3 27B",
    "MedGemma 27B",
    "Mistral Small 3.2 24B",
]

SHORT = {
    "Gemma 3 4B": "Gemma 3 4B",
    "MedGemma 4B": "MedGemma 4B",
    "Ministral 3B": "Ministral 3B",
    "Gemma 3 27B": "Gemma 3 27B",
    "MedGemma 27B": "MedGemma 27B",
    "Mistral Small 3.2 24B": "Mistral 24B",
}

DISPLAY = {
    ("Gemma 3 4B", "Zero-shot"): "Gemma 3 4B",
    ("MedGemma 4B", "Zero-shot"): "MedGemma 4B",
    ("Ministral 3B", "Zero-shot"): "Ministral 3B zero-shot",
    ("Gemma 3 27B", "Zero-shot"): "Gemma 3 27B",
    ("MedGemma 27B", "Zero-shot"): "MedGemma 27B",
    ("Mistral Small 3.2 24B", "Zero-shot"): "Mistral 24B zero-shot",
    ("Gemma 3 4B", "QLoRA"): "Gemma 3 4B QLoRA",
    ("MedGemma 4B", "QLoRA"): "MedGemma 4B QLoRA",
    ("Ministral 3B", "QLoRA"): "Ministral 3B QLoRA",
    ("Gemma 3 27B", "QLoRA"): "Gemma 3 27B QLoRA",
    ("MedGemma 27B", "QLoRA"): "MedGemma 27B QLoRA",
    ("Mistral Small 3.2 24B", "QLoRA"): "Mistral 24B QLoRA",
}

SCALE = [
    ("Gemma 3", "Gemma 3 27B", "Gemma 3 4B", "27B vs 4B"),
    ("MedGemma", "MedGemma 27B", "MedGemma 4B", "27B vs 4B"),
    ("Mistral", "Mistral Small 3.2 24B", "Ministral 3B", "24B vs 3B"),
]

validation = pd.read_csv(FINAL / "val_cd1to5.csv", low_memory=False)
validation_ids = np.arange(len(validation))
cleaned_ids = validation_ids[
    validation["cr_text_cleaned"].fillna("").astype(str).str.strip().ne("").to_numpy()
]

configs = {}

for model, series, size, method, directory, schema in SOURCES:
    for path in sorted(directory.rglob("paper_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))

        if schema == "llm":
            text = "cleaned" if "cleaned" in summary["text_col"] else "raw"
            strategy = summary["mode"]
        else:
            text = summary["text_mode"]
            strategy = summary["target_mode"]

        patient = pd.read_csv(path.parent / "val_predictions_patient_level.csv")
        report = pd.read_csv(path.parent / "val_predictions_cr_level.csv")

        patient["patient_id"] = patient["patient_id"].astype(str)
        patient["cd_manuel"] = patient["cd_manuel"].astype(int)
        patient["cd_pred"] = patient["cd_pred"].astype(int)

        report["patient_id"] = report["patient_id"].astype(str)
        report["cd_manuel"] = report["cd_manuel"].astype(int)
        report["cd_pred"] = report["cd_pred"].astype(int)
        report["_report_id"] = (
            cleaned_ids
            if schema == "llm" and text == "cleaned"
            else validation_ids
        )

        configs[(model, method, text, strategy)] = {
            "patient": patient,
            "report": report,
        }

comparisons = []
comparison_order = 0

for strategy, family, column in [
    ("direct", "1a Fine-tuning, direct", "One-step"),
    ("twostep", "1b Fine-tuning, two-step", "Two-step"),
]:
    for model in MODEL_ORDER:
        for text in ["raw", "cleaned"]:
            comparisons.append(
                (
                    "a",
                    "Fine-tuning (QLoRA vs zero-shot)",
                    family,
                    column,
                    f"{SHORT[model]} {text}, QLoRA vs zero-shot",
                    (model, "QLoRA", text, strategy),
                    (model, "Zero-shot", text, strategy),
                    comparison_order,
                )
            )
            comparison_order += 1

for panel, method, families, title in [
    (
        "b",
        "Zero-shot",
        [
            ("direct", "5a Scale zero-shot, direct", "One-step"),
            ("twostep", "5b Scale zero-shot, two-step", "Two-step"),
        ],
        "Model scale, zero-shot (large vs small)",
    ),
    (
        "c",
        "QLoRA",
        [
            ("direct", "5c Scale QLoRA, direct", "One-step"),
            ("twostep", "5d Scale QLoRA, two-step", "Two-step"),
        ],
        "Model scale, QLoRA (large vs small)",
    ),
]:
    for strategy, family, column in families:
        for series, large, small, size_label in SCALE:
            for text in ["raw", "cleaned"]:
                comparisons.append(
                    (
                        panel,
                        title,
                        family,
                        column,
                        f"{series}, {size_label}, {text}",
                        (large, method, text, strategy),
                        (small, method, text, strategy),
                        comparison_order,
                    )
                )
                comparison_order += 1

for strategy, family, column in [
    ("direct", "8a Small QLoRA vs large zero-shot, direct", "One-step"),
    ("twostep", "8b Small QLoRA vs large zero-shot, two-step", "Two-step"),
]:
    for series, large, small, size_label in SCALE:
        for text in ["raw", "cleaned"]:
            comparisons.append(
                (
                    "d",
                    "Model scale and fine-tuning (small QLoRA vs large zero-shot)",
                    family,
                    column,
                    f"{series}, {text}",
                    (small, "QLoRA", text, strategy),
                    (large, "Zero-shot", text, strategy),
                    comparison_order,
                )
            )
            comparison_order += 1

for text, family, column in [
    ("raw", "2a Two-step vs direct, raw", "Raw text"),
    ("cleaned", "2b Two-step vs direct, cleaned", "Cleaned text"),
]:
    for model in MODEL_ORDER:
        for method in ["Zero-shot", "QLoRA"]:
            comparisons.append(
                (
                    "e",
                    "Prompting strategy (two-step vs one-step)",
                    family,
                    column,
                    DISPLAY[(model, method)],
                    (model, method, text, "twostep"),
                    (model, method, text, "direct"),
                    comparison_order,
                )
            )
            comparison_order += 1

for strategy, family, column in [
    ("direct", "3a Raw vs cleaned, direct", "One-step"),
    ("twostep", "3b Raw vs cleaned, two-step", "Two-step"),
]:
    for model in MODEL_ORDER:
        for method in ["Zero-shot", "QLoRA"]:
            comparisons.append(
                (
                    "f",
                    "Text preprocessing (raw vs cleaned)",
                    family,
                    column,
                    DISPLAY[(model, method)],
                    (model, method, "raw", strategy),
                    (model, method, "cleaned", strategy),
                    comparison_order,
                )
            )
            comparison_order += 1

G_SPECS = [
    ("Zero-shot", "Mistral Small 3.2 24B", "MedGemma 27B"),
    ("Zero-shot", "Mistral Small 3.2 24B", "Gemma 3 27B"),
    ("Zero-shot", "Ministral 3B", "MedGemma 4B"),
    ("Zero-shot", "Ministral 3B", "Gemma 3 4B"),
    ("QLoRA", "Mistral Small 3.2 24B", "MedGemma 27B"),
    ("QLoRA", "Ministral 3B", "MedGemma 4B"),
    ("QLoRA", "Ministral 3B", "Gemma 3 4B"),
    ("QLoRA", "Mistral Small 3.2 24B", "Gemma 3 27B"),
]

for strategy, family, column in [
    ("direct", "6a Model family, direct", "One-step"),
    ("twostep", "6b Model family, two-step", "Two-step"),
]:
    for method, first_model, second_model in G_SPECS:
        for text in ["raw", "cleaned"]:
            comparisons.append(
                (
                    "g",
                    "Model family",
                    family,
                    column,
                    f"{method} {text}, {SHORT[first_model]} vs {SHORT[second_model]}",
                    (first_model, method, text, strategy),
                    (second_model, method, text, strategy),
                    comparison_order,
                )
            )
            comparison_order += 1

H_SPECS = [
    ("Zero-shot", "MedGemma 27B", "Gemma 3 27B", "27B"),
    ("Zero-shot", "MedGemma 4B", "Gemma 3 4B", "4B"),
    ("QLoRA", "MedGemma 4B", "Gemma 3 4B", "4B"),
    ("QLoRA", "MedGemma 27B", "Gemma 3 27B", "27B"),
]

for strategy, family, column in [
    ("direct", "7a Medical pretraining, direct", "One-step"),
    ("twostep", "7b Medical pretraining, two-step", "Two-step"),
]:
    for method, first_model, second_model, size_label in H_SPECS:
        for text in ["raw", "cleaned"]:
            comparisons.append(
                (
                    "h",
                    "Medical pretraining (MedGemma vs Gemma 3)",
                    family,
                    column,
                    f"{method} {size_label} {text}, MedGemma vs Gemma 3",
                    (first_model, method, text, strategy),
                    (second_model, method, text, strategy),
                    comparison_order,
                )
            )
            comparison_order += 1

rng_patient = np.random.default_rng(SEED)
rng_report = np.random.default_rng(SEED)

first_config = next(iter(configs.values()))
patient_ids = sorted(first_config["patient"]["patient_id"].unique())
patient_samples = rng_patient.integers(
    0,
    len(patient_ids),
    size=(B, len(patient_ids)),
)

rows = []

for panel, panel_title, family, column, label, first, second, order in comparisons:
    patient_a = configs[first]["patient"].set_index("patient_id").loc[patient_ids]
    patient_b = configs[second]["patient"].set_index("patient_id").loc[patient_ids]

    gold = patient_a["cd_manuel"].to_numpy(dtype=int)
    pred_a = patient_a["cd_pred"].to_numpy(dtype=int)
    pred_b = patient_b["cd_pred"].to_numpy(dtype=int)

    kappa_a = cohen_kappa_score(gold, pred_a, labels=LABELS, weights="quadratic")
    kappa_b = cohen_kappa_score(gold, pred_b, labels=LABELS, weights="quadratic")

    differences = []

    for indices in patient_samples:
        resampled_gold = gold[indices]

        if np.unique(resampled_gold).size < 2:
            continue

        differences.append(
            cohen_kappa_score(
                resampled_gold,
                pred_a[indices],
                labels=LABELS,
                weights="quadratic",
            )
            - cohen_kappa_score(
                resampled_gold,
                pred_b[indices],
                labels=LABELS,
                weights="quadratic",
            )
        )

    differences = np.asarray(differences)
    p_raw = (1 + np.sum(differences <= 0)) / (1 + differences.size)
    p_raw = 2 * min(p_raw, 1 - p_raw)

    rows.append(
        {
            "level": "patient",
            "panel": panel,
            "panel_title": panel_title,
            "family": family,
            "column": column,
            "comparison": label,
            "kappa_a": round(float(kappa_a), 3),
            "kappa_b": round(float(kappa_b), 3),
            "difference": round(float(kappa_a - kappa_b), 3),
            "ci_low": round(float(np.percentile(differences, 2.5)), 3),
            "ci_high": round(float(np.percentile(differences, 97.5)), 3),
            "p_raw": round(float(p_raw), 4),
            "n_bootstrap": int(differences.size),
            "_order": order,
        }
    )

    report_a = configs[first]["report"].set_index("_report_id")
    report_b = configs[second]["report"].set_index("_report_id")

    paired = report_a[["patient_id", "cd_manuel", "cd_pred"]].join(
        report_b[["cd_manuel", "cd_pred"]],
        how="inner",
        lsuffix="_a",
        rsuffix="_b",
    )

    gold = paired["cd_manuel_a"].to_numpy(dtype=int)
    pred_a = paired["cd_pred_a"].to_numpy(dtype=int)
    pred_b = paired["cd_pred_b"].to_numpy(dtype=int)
    patient_values = paired["patient_id"].to_numpy()
    unique_patients = pd.unique(patient_values)
    indices_by_patient = [
        np.flatnonzero(patient_values == patient_id)
        for patient_id in unique_patients
    ]

    kappa_a = cohen_kappa_score(gold, pred_a, labels=LABELS, weights="quadratic")
    kappa_b = cohen_kappa_score(gold, pred_b, labels=LABELS, weights="quadratic")

    differences = []

    for _ in range(B):
        sampled = rng_report.integers(
            0,
            len(unique_patients),
            size=len(unique_patients),
        )
        indices = np.concatenate(
            [indices_by_patient[index] for index in sampled]
        )
        resampled_gold = gold[indices]

        if np.unique(resampled_gold).size < 2:
            continue

        differences.append(
            cohen_kappa_score(
                resampled_gold,
                pred_a[indices],
                labels=LABELS,
                weights="quadratic",
            )
            - cohen_kappa_score(
                resampled_gold,
                pred_b[indices],
                labels=LABELS,
                weights="quadratic",
            )
        )

    differences = np.asarray(differences)
    p_raw = (1 + np.sum(differences <= 0)) / (1 + differences.size)
    p_raw = 2 * min(p_raw, 1 - p_raw)

    rows.append(
        {
            "level": "report",
            "panel": panel,
            "panel_title": panel_title,
            "family": family,
            "column": column,
            "comparison": label,
            "kappa_a": round(float(kappa_a), 3),
            "kappa_b": round(float(kappa_b), 3),
            "difference": round(float(kappa_a - kappa_b), 3),
            "ci_low": round(float(np.percentile(differences, 2.5)), 3),
            "ci_high": round(float(np.percentile(differences, 97.5)), 3),
            "p_raw": round(float(p_raw), 4),
            "n_bootstrap": int(differences.size),
            "_order": order,
        }
    )

results = pd.DataFrame(rows)
results["p_holm"] = np.nan

for level in ["patient", "report"]:
    for family in results.loc[results["level"] == level, "family"].unique():
        block = results.loc[
            (results["level"] == level)
            & (results["family"] == family)
        ].sort_values("p_raw")
        m = len(block)
        adjusted = 0.0

        for rank, (index, row) in enumerate(block.iterrows()):
            adjusted = max(adjusted, (m - rank) * row["p_raw"])
            results.loc[index, "p_holm"] = round(min(adjusted, 1.0), 4)

results["significant"] = np.where(results["p_holm"] < 0.05, "*", "")

results[
    [
        "level",
        "panel",
        "panel_title",
        "family",
        "column",
        "comparison",
        "kappa_a",
        "kappa_b",
        "difference",
        "ci_low",
        "ci_high",
        "p_raw",
        "p_holm",
        "significant",
        "n_bootstrap",
    ]
].to_csv(
    OUT / "Pairwise-comparisons.csv",
    index=False,
    encoding="utf-8-sig",
)

PANELS = [
    {
        "letter": "a",
        "title": "Fine-tuning (QLoRA vs zero-shot)",
        "left": ("1a Fine-tuning, direct", "One-step"),
        "right": ("1b Fine-tuning, two-step", "Two-step"),
        "strip": [", QLoRA vs zero-shot"],
        "rename": {},
    },
    {
        "letter": "b",
        "title": "Model scale, zero-shot (large vs small)",
        "left": ("5a Scale zero-shot, direct", "One-step"),
        "right": ("5b Scale zero-shot, two-step", "Two-step"),
        "strip": [", 27B vs 4B"],
        "rename": {"Mistral, 24B vs 3B": "Mistral 24B vs Ministral 3B"},
    },
    {
        "letter": "c",
        "title": "Model scale, QLoRA (large vs small)",
        "left": ("5c Scale QLoRA, direct", "One-step"),
        "right": ("5d Scale QLoRA, two-step", "Two-step"),
        "strip": [", 27B vs 4B"],
        "rename": {"Mistral, 24B vs 3B": "Mistral 24B vs Ministral 3B"},
    },
    {
        "letter": "d",
        "title": "Model scale and fine-tuning (small QLoRA vs large zero-shot)",
        "left": ("8a Small QLoRA vs large zero-shot, direct", "One-step"),
        "right": ("8b Small QLoRA vs large zero-shot, two-step", "Two-step"),
        "strip": [],
        "rename": {"Mistral,": "Ministral 3B vs Mistral 24B,"},
    },
    {
        "letter": "e",
        "title": "Prompting strategy (two-step vs one-step)",
        "left": ("2a Two-step vs direct, raw", "Raw text"),
        "right": ("2b Two-step vs direct, cleaned", "Cleaned text"),
        "strip": [],
        "rename": {},
    },
    {
        "letter": "f",
        "title": "Text preprocessing (raw vs cleaned)",
        "left": ("3a Raw vs cleaned, direct", "One-step"),
        "right": ("3b Raw vs cleaned, two-step", "Two-step"),
        "strip": [],
        "rename": {},
    },
    {
        "letter": "g",
        "title": "Model family",
        "left": ("6a Model family, direct", "One-step"),
        "right": ("6b Model family, two-step", "Two-step"),
        "strip": [],
        "rename": {},
    },
    {
        "letter": "h",
        "title": "Medical pretraining (MedGemma vs Gemma 3)",
        "left": ("7a Medical pretraining, direct", "One-step"),
        "right": ("7b Medical pretraining, two-step", "Two-step"),
        "strip": [", MedGemma vs Gemma 3"],
        "rename": {},
    },
]

COLOR = "#2E5A88"
COLOR_SIGNIFICANT = "#0B4F8A"
FIGURE_WIDTH = 13.0
ROW_HEIGHT = 0.22
PANEL_GAP = 0.85
TOP_MARGIN = 0.35
BOTTOM_MARGIN = 0.55
LABEL_PAD = 62
PANEL_WIDTH = 0.245
LEFT_START = 0.255
RIGHT_START = 0.665

for level, figure_name in [
    ("patient", "Figure 3 - forest plots patient-level"),
    ("report", "Figure S6 - forest plots report-level"),
]:
    forest = results.loc[results["level"] == level].copy()
    all_low = forest["ci_low"].min()
    all_high = forest["ci_high"].max()
    margin = 0.06 * (all_high - all_low)
    x_low = all_low - margin
    x_high = all_high + margin

    total_lines = 0

    for panel in PANELS:
        family = panel["left"][0]
        total_lines += forest.loc[forest["family"] == family, "comparison"].nunique()

    figure_height = (
        TOP_MARGIN
        + BOTTOM_MARGIN
        + ROW_HEIGHT * total_lines
        + PANEL_GAP * len(PANELS)
    )

    figure = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    cursor = figure_height - TOP_MARGIN

    for panel in PANELS:
        left_family, left_title = panel["left"]
        right_family, right_title = panel["right"]

        left = (
            forest.loc[forest["family"] == left_family]
            .sort_values("_order")
            .set_index("comparison")
        )
        right = (
            forest.loc[forest["family"] == right_family]
            .sort_values("_order")
            .set_index("comparison")
        )

        labels = list(left.index)
        labels += [label for label in right.index if label not in labels]

        display = []

        for label in labels:
            text = label.replace("direct", "one-step")

            for source, target in panel["rename"].items():
                text = text.replace(source, target)

            for pattern in panel["strip"]:
                text = text.replace(pattern, "")

            display.append(text.strip().strip(",").strip())

        n_lines = len(labels)
        positions = np.arange(n_lines)[::-1]
        panel_height = ROW_HEIGHT * n_lines
        cursor -= PANEL_GAP
        bottom = (cursor - panel_height) / figure_height
        height = panel_height / figure_height

        figure.text(
            0.012,
            (cursor + 0.30) / figure_height,
            f'{panel["letter"]}. {panel["title"]}',
            fontsize=11,
            family="Serif",
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        axes = []

        for column_index, (column_title, block) in enumerate(
            [(left_title, left), (right_title, right)]
        ):
            left_position = LEFT_START if column_index == 0 else RIGHT_START
            axis = figure.add_axes(
                [left_position, bottom, PANEL_WIDTH, height]
            )
            axes.append(axis)

            axis.axhspan(
                -0.5,
                n_lines - 0.5,
                color="#000000",
                alpha=0.035,
                zorder=0,
            )
            axis.axvline(
                0,
                color="#808080",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
            )

            for position, label in zip(positions, labels):
                if label not in block.index:
                    continue

                entry = block.loc[label]
                significant = float(entry["p_holm"]) < 0.05

                axis.errorbar(
                    entry["difference"],
                    position,
                    xerr=[
                        [entry["difference"] - entry["ci_low"]],
                        [entry["ci_high"] - entry["difference"]],
                    ],
                    fmt="D",
                    markersize=4.4,
                    markerfacecolor=COLOR_SIGNIFICANT if significant else "#FFFFFF",
                    markeredgecolor=COLOR,
                    markeredgewidth=1.0,
                    color=COLOR,
                    ecolor=COLOR,
                    capsize=1.8,
                    elinewidth=0.9,
                    zorder=3,
                )

                axis.annotate(
                    f'{entry["difference"]:+.3f}',
                    xy=(-0.03, position),
                    xycoords=("axes fraction", "data"),
                    fontsize=7.5,
                    family="Serif",
                    ha="right",
                    va="center",
                    annotation_clip=False,
                )

                p_value = float(entry["p_holm"])
                p_text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"

                if significant:
                    p_text += " *"
                    axis.annotate(
                        "*",
                        xy=(entry["ci_high"], position),
                        xytext=(3, -1),
                        textcoords="offset points",
                        fontsize=9,
                        family="Serif",
                        va="center",
                        annotation_clip=False,
                    )

                axis.annotate(
                    p_text,
                    xy=(1.03, position),
                    xycoords=("axes fraction", "data"),
                    fontsize=7.5,
                    family="Serif",
                    fontweight="bold" if significant else "normal",
                    ha="left",
                    va="center",
                    annotation_clip=False,
                )

            axis.set_xlim(x_low, x_high)
            axis.set_ylim(-0.7, n_lines - 0.3)
            axis.set_yticks(positions)
            axis.tick_params(axis="y", length=0, pad=LABEL_PAD)
            axis.tick_params(axis="x", labelsize=7.5)
            axis.grid(axis="x", alpha=0.22)
            axis.set_axisbelow(True)

            for spine in ["top", "right", "left"]:
                axis.spines[spine].set_visible(False)

            axis.set_title(
                column_title,
                fontsize=9.5,
                family="Serif",
                pad=14,
            )

            axis.annotate(
                r"$\Delta \ K_{\mathrm{QW}}$",
                xy=(-0.03, 1.0),
                xytext=(0, 4),
                xycoords="axes fraction",
                textcoords="offset points",
                fontsize=8,
                family="Serif",
                fontweight="bold",
                ha="right",
                va="bottom",
                annotation_clip=False,
            )

            axis.annotate(
                "Holm's corrected\np-value",
                xy=(1.03, 1.0),
                xytext=(0, 2),
                xycoords="axes fraction",
                textcoords="offset points",
                fontsize=7,
                family="Serif",
                fontweight="bold",
                ha="left",
                va="bottom",
                annotation_clip=False,
            )

            if column_index == 1:
                axis.set_yticklabels([])

        axes[0].set_yticklabels(
            display,
            fontsize=7.5,
            family="Serif",
        )

        cursor -= panel_height

    figure.savefig(
        OUT / f"{figure_name}.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUT / f"{figure_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
