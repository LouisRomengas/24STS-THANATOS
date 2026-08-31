import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DOCX = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_Resultats-annotations"
    "/train_val_entities/Tables_complications_characteristics.docx"
)

OUTPUT = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results/figures"
)

GROUPS = [
    ("Non-LLM classifiers", [
        "TF-IDF",
        "Qwen3 Embeddings 8B",
    ]),
    ("Zero-shot LLMs (small scale)", [
        "Gemma 3 4B",
        "MedGemma 4B",
        "Ministral 3B zero-shot",
    ]),
    ("Zero-shot LLMs (large scale)", [
        "Gemma 3 27B",
        "MedGemma 27B",
        "Mistral Small 3.2 24B zero-shot",
    ]),
    ("QLoRA fine-tuned LLMs (small scale)", [
        "Gemma 3 4B QLoRA",
        "MedGemma 4B QLoRA",
        "Ministral 3B QLoRA",
    ]),
    ("QLoRA fine-tuned LLMs (large scale)", [
        "Gemma 3 27B QLoRA",
        "MedGemma 27B QLoRA",
        "Mistral Small 3.2 24B QLoRA",
    ]),
]

SHORT_NAMES = {
    "TF-IDF": "TF-IDF (LR)",
    "Qwen3 Embeddings 8B": "Qwen3-Embedding (LR)",
    "Gemma 3 4B": "Gemma 3 4B",
    "Gemma 3 4B QLoRA": "Gemma 3 4B",
    "MedGemma 4B": "MedGemma 4B",
    "MedGemma 4B QLoRA": "MedGemma 4B",
    "Ministral 3B zero-shot": "Ministral 3B",
    "Ministral 3B QLoRA": "Ministral 3B",
    "Gemma 3 27B": "Gemma 3 27B",
    "Gemma 3 27B QLoRA": "Gemma 3 27B",
    "MedGemma 27B": "MedGemma 27B",
    "MedGemma 27B QLoRA": "MedGemma 27B",
    "Mistral Small 3.2 24B zero-shot": "Mistral Small 3.2 24B",
    "Mistral Small 3.2 24B QLoRA": "Mistral Small 3.2 24B",
}

RECALL_COLUMNS = [
    ("No complications", 1, "sans"),
    ("CD-I", 1, "avec"),
    ("CD-II", 2, None),
    ("CD-III", 3, None),
    ("CD-IV", 4, None),
    ("CD-V", 5, None),
]

MAE_COLUMNS = [
    ("Overall", "overall", None),
    ("0", "complication", "0"),
    ("1–2", "complication", "1-2"),
    ("≥3", "complication", "3+"),
    ("Short", "length", "short"),
    ("Long", "length", "long"),
]

MAE_SUPER_HEADERS = [
    ("Complication burden", 1, 3),
    ("Report length", 4, 5),
]

MAE_CAP = 1.0

CMAP_RECALL = LinearSegmentedColormap.from_list(
    "recall",
    ["#FFFFFF", "#9DC3E6", "#2E5A88", "#0B4F8A"],
)
CMAP_RECALL.set_bad("#FFFFFF")

CMAP_MAE = LinearSegmentedColormap.from_list(
    "mae",
    ["#FFFFFF", "#F5C99B", "#E8830C", "#B04A00"],
)
CMAP_MAE.set_bad("#FFFFFF")

cr_level = cr_dataset.copy()

cr_level["n_complications"] = cr_level["n_complications"].fillna(0)

cr_level = cr_level.loc[
    ~(
        (cr_level["text_mode"] == "cleaned")
        & (cr_level["n_tokens_edsnlp"] == 0)
    )
].copy()

patient_agg = (
    cr_level
    .groupby(
        [
            "model_order",
            "model",
            "configuration",
            "text_mode",
            "target_mode",
            "patient_id",
        ],
        as_index=False,
    )
    .agg(
        cd_manuel=("gold", "max"),
        cd_pred=("pred", "max"),
        n_tokens_edsnlp=("n_tokens_edsnlp", "sum"),
        n_complications=("n_complications", "first"),
    )
)

patient_agg["absolute_error"] = (
    patient_agg["cd_pred"] - patient_agg["cd_manuel"]
).abs()

patient_agg["complication_stratum"] = pd.cut(
    patient_agg["n_complications"],
    bins=[-1, 0, 2, np.inf],
    labels=["0", "1-2", "3+"],
)

cutoff = patient_agg["text_mode"].map(
    {
        "raw": 2148.0,
        "cleaned": 337.0,
    }
)

patient_agg["length_stratum"] = pd.Categorical(
    np.where(
        patient_agg["n_tokens_edsnlp"] <= cutoff,
        "short",
        "long",
    ),
    categories=["short", "long"],
    ordered=True,
)

with zipfile.ZipFile(DOCX) as archive:
    xml = archive.read("word/document.xml").decode("utf-8")

texte = re.sub(r"<w:p[ >]", "\n<w:p ", xml)
texte = re.sub(r"<[^>]+>", " ", texte)
texte = texte.replace("&amp;", "&")

paires = re.findall(r"(P[0-9a-f]{8})\s+(\d+)\s", texte)

vus = {}

for patient_id, n_complications in paires:
    if patient_id not in vus:
        vus[patient_id] = int(n_complications)

avec_complication = set(vus)

selected = results_patient["ranking"].loc[
    results_patient["ranking"]["retenue"] == "*"
].copy()

kappa_by_model = dict(
    zip(
        selected["model"],
        selected["kappa_quadratic"],
    )
)

config_by_model = dict(
    zip(
        selected["model"],
        selected["configuration"],
    )
)

frame = patient_agg.merge(
    selected[["model", "configuration"]],
    on=["model", "configuration"],
    how="inner",
)

frame["cd1_groupe"] = np.where(
    frame["patient_id"].astype(str).str.strip().isin(avec_complication),
    "avec",
    "sans",
)

rows = []

for group_index, (group_name, models) in enumerate(GROUPS):
    if group_index > 0:
        rows.append(
            {
                "kind": "gap",
                "label": "",
                "group": None,
            }
        )

    rows.append(
        {
            "kind": "gap",
            "label": "",
            "group": group_name,
        }
    )

    for model in models:
        if model not in config_by_model:
            continue

        configuration = config_by_model[model]

        block = frame.loc[
            (frame["model"] == model)
            & (frame["configuration"] == configuration)
        ]

        if block.empty:
            continue

        recalls = []

        for _, grade, groupe in RECALL_COLUMNS:
            reference = block.loc[
                block["cd_manuel"] == grade
            ]

            if groupe is not None:
                reference = reference.loc[
                    reference["cd1_groupe"] == groupe
                ]

            recalls.append(
                float(
                    (
                        reference["cd_pred"] == grade
                    ).mean()
                )
                if len(reference)
                else np.nan
            )

        maes = []

        for _, kind, value in MAE_COLUMNS:
            if kind == "overall":
                subset = block
            elif kind == "complication":
                subset = block.loc[
                    block["complication_stratum"] == value
                ]
            else:
                subset = block.loc[
                    block["length_stratum"] == value
                ]

            maes.append(
                float(subset["absolute_error"].mean())
                if len(subset)
                else np.nan
            )

        rows.append(
            {
                "kind": "model",
                "label": "{}   $k_{{\\mathrm{{QW}}}}$ {:.3f}".format(
                    SHORT_NAMES.get(model, model),
                    kappa_by_model[model],
                ),
                "group": None,
                "recalls": recalls,
                "maes": maes,
            }
        )

n_rows = len(rows)

model_indices = [
    index
    for index, row in enumerate(rows)
    if row["kind"] == "model"
]

reference_model = GROUPS[0][1][0]

reference_block = frame.loc[
    (frame["model"] == reference_model)
    & (
        frame["configuration"]
        == config_by_model[reference_model]
    )
]

supports = []

for _, grade, groupe in RECALL_COLUMNS:
    reference = reference_block.loc[
        reference_block["cd_manuel"] == grade
    ]

    if groupe is not None:
        reference = reference.loc[
            reference["cd1_groupe"] == groupe
        ]

    supports.append(len(reference))

mae_supports = []

for _, kind, value in MAE_COLUMNS:
    if kind == "overall":
        mae_supports.append(len(reference_block))
    elif kind == "complication":
        mae_supports.append(
            int(
                (
                    reference_block["complication_stratum"] == value
                ).sum()
            )
        )
    else:
        mae_supports.append(
            int(
                (
                    reference_block["length_stratum"] == value
                ).sum()
            )
        )

recall_matrix = np.full(
    (n_rows, len(RECALL_COLUMNS)),
    np.nan,
)

mae_matrix = np.full(
    (n_rows, len(MAE_COLUMNS)),
    np.nan,
)

for index, row in enumerate(rows):
    if row["kind"] != "model":
        continue

    recall_matrix[index] = row["recalls"]
    mae_matrix[index] = row["maes"]

fig = plt.figure(
    figsize=(16.5, 0.34 * n_rows + 2.1)
)

grid = fig.add_gridspec(
    1,
    5,
    width_ratios=[1.2, 0.026, 0.20, 1.2, 0.026],
    wspace=0.10,
    left=0.20,
    right=0.94,
    top=0.86,
    bottom=0.03,
)

ax_recall = fig.add_subplot(grid[0, 0])
ax_recall_bar = fig.add_subplot(grid[0, 1])
ax_mae = fig.add_subplot(grid[0, 3])
ax_mae_bar = fig.add_subplot(grid[0, 4])

image_recall = ax_recall.imshow(
    recall_matrix,
    cmap=CMAP_RECALL,
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
)

ax_recall.set_xticks(
    np.arange(len(RECALL_COLUMNS))
)

ax_recall.set_xticklabels(
    [
        "{}\n(n={})".format(name, count)
        for (name, _, _), count in zip(
            RECALL_COLUMNS,
            supports,
        )
    ],
    fontsize=8.5,
    family="Serif",
)

ax_recall.xaxis.set_ticks_position("top")
ax_recall.tick_params(axis="x", pad=3)

ax_recall.annotate(
    "",
    xy=(-0.42, -0.5),
    xytext=(5.42, -0.5),
    xycoords=("data", "data"),
    textcoords=("data", "data"),
    arrowprops={
        "arrowstyle": "-",
        "linewidth": 0.8,
        "color": "#666666",
        "shrinkA": 0,
        "shrinkB": 0,
    },
    annotation_clip=False,
)

ax_recall.set_yticks(
    np.arange(n_rows)
)

ax_recall.set_yticklabels(
    [row["label"] for row in rows],
    fontsize=9,
    family="Serif",
)

image_mae = ax_mae.imshow(
    np.clip(
        mae_matrix,
        0.0,
        MAE_CAP,
    ),
    cmap=CMAP_MAE,
    vmin=0.0,
    vmax=MAE_CAP,
    aspect="auto",
)

ax_mae.set_xticks(
    np.arange(len(MAE_COLUMNS))
)

ax_mae.set_xticklabels(
    [
        "{}\n(n={})".format(name, count)
        for (name, _, _), count in zip(
            MAE_COLUMNS,
            mae_supports,
        )
    ],
    fontsize=9,
    family="Serif",
)

ax_mae.xaxis.set_ticks_position("top")
ax_mae.set_yticks([])

for title, first, last in MAE_SUPER_HEADERS:
    centre = (first + last) / 2.0

    ax_mae.annotate(
        title,
        xy=(centre, -0.5),
        xytext=(0, 40),
        xycoords=("data", "data"),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
        family="Serif",
        fontweight="bold",
        annotation_clip=False,
    )

    ax_mae.annotate(
        "",
        xy=(first - 0.42, -0.5),
        xytext=(last + 0.42, -0.5),
        xycoords=("data", "data"),
        textcoords=("data", "data"),
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.8,
            "color": "#666666",
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )

for index, row in enumerate(rows):
    if row["kind"] == "model":
        for column in range(len(RECALL_COLUMNS)):
            value = recall_matrix[index, column]

            if not np.isnan(value):
                ax_recall.text(
                    column,
                    index,
                    "{:.2f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    family="Serif",
                    color=(
                        "#FFFFFF"
                        if value > 0.55
                        else "#333333"
                    ),
                )

        for column in range(len(MAE_COLUMNS)):
            value = mae_matrix[index, column]

            if not np.isnan(value):
                ax_mae.text(
                    column,
                    index,
                    "{:.2f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    family="Serif",
                    color=(
                        "#FFFFFF"
                        if min(value, MAE_CAP) > 0.60
                        else "#333333"
                    ),
                )

    elif row["group"] is not None:
        ax_recall.text(
            -0.52,
            index,
            row["group"],
            ha="left",
            va="center",
            fontsize=9.5,
            family="Serif",
            fontweight="bold",
            color="#333333",
        )

for axis, n_columns in (
    (ax_recall, len(RECALL_COLUMNS)),
    (ax_mae, len(MAE_COLUMNS)),
):
    axis.set_xticks(
        np.arange(-0.5, n_columns, 1),
        minor=True,
    )

    axis.set_yticks([], minor=True)

    axis.grid(
        which="minor",
        axis="x",
        color="#FFFFFF",
        linewidth=1.4,
    )

    axis.tick_params(
        which="minor",
        length=0,
    )

    axis.tick_params(
        which="major",
        length=0,
    )

    for index in model_indices:
        if index + 1 in model_indices:
            axis.axhline(
                index + 0.5,
                color="#FFFFFF",
                linewidth=1.4,
                zorder=3,
            )

    for spine in axis.spines.values():
        spine.set_visible(False)

bar_recall = fig.colorbar(
    image_recall,
    cax=ax_recall_bar,
)

bar_recall.outline.set_visible(False)

bar_recall.ax.tick_params(
    labelsize=8,
    length=2,
    pad=3,
)

bar_recall.set_ticks(
    [0.0, 0.5, 1.0]
)

bar_mae = fig.colorbar(
    image_mae,
    cax=ax_mae_bar,
)

bar_mae.outline.set_visible(False)

bar_mae.ax.tick_params(
    labelsize=8,
    length=2,
    pad=3,
)

bar_mae.set_ticks(
    [0.0, 0.5, 1.0]
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT / "Figure 4 - heatmap patient-level.pdf",
    bbox_inches="tight",
)

plt.close(fig)
