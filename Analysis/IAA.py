import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BRAT = Path("/export/home/cse240018/brat_data/24STS-THANATOS/Calibration_annotateurs")
RESULTS = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results/Performances")
ANNOTATEURS = ["ACE", "ADE", "CRE", "STS"]
GRADES = ["1", "2", "3a", "3b", "4a", "4b", "5"]
RANG = {grade: index for index, grade in enumerate(GRADES)}
TRAITEMENTS_CD3 = {
    "TRAITEMENT_CHIRURGICAL",
    "TRAITEMENT_RADIOLOGIQUE",
    "TRAITEMENT_ENDOSCOPIQUE",
}
N_BOOTSTRAP = 5000
GRAINE = 42

RESULTS.mkdir(parents=True, exist_ok=True)

scores = defaultdict(lambda: defaultdict(lambda: "1"))

for annotateur in ANNOTATEURS:
    base = BRAT / annotateur

    for path in sorted(base.rglob("*.ann")):
        if ".ipynb_checkpoints" in path.parts or "__pycache__" in path.parts:
            continue

        relative = path.relative_to(base).parts

        if len(relative) < 2:
            continue

        patient_id = relative[0]
        entities = {}
        attributes = defaultdict(dict)

        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")

            if len(fields) < 2:
                continue

            if fields[0].startswith("T"):
                entities[fields[0]] = fields[1].split(" ", 1)[0]

            elif fields[0].startswith("A"):
                values = fields[1].split(" ")

                if len(values) >= 3:
                    attributes[values[1]][values[0]] = values[2]

        grade = "1"

        for entity_id, entity_type in entities.items():
            attrs = attributes.get(entity_id, {})
            candidate = None

            if entity_type == "DECES":
                candidate = "5"

            elif entity_type == "TRAITEMENT_REANIMATION":
                value = attrs.get("CD_Type_Defaillance") or attrs.get("CD_Sous_Grade")
                candidate = "4b" if value == "b" else "4a"

            elif entity_type in TRAITEMENTS_CD3:
                value = attrs.get("CD_Sous_Grade") or attrs.get("CD_Type_Defaillance")
                candidate = "3b" if value == "b" else "3a"

            elif entity_type == "TRAITEMENT_MEDICAL":
                candidate = "2" if attrs.get("CD_Grade") in ("II", "2") else "1"

            if candidate is not None and RANG[candidate] > RANG[grade]:
                grade = candidate

        if RANG[grade] > RANG[scores[patient_id][annotateur]]:
            scores[patient_id][annotateur] = grade

pivot = (
    pd.DataFrame.from_dict(scores, orient="index")
    .reindex(columns=ANNOTATEURS)
    .sort_index()
    .dropna()
)

data = pivot.to_numpy()
observed = [grade for grade in GRADES if (data == grade).any()]
q = len(observed)
category_index = {grade: index for index, grade in enumerate(observed)}
codes = np.array(
    [[category_index[value] for value in row] for row in data],
    dtype=int,
)
one_hot = np.eye(q, dtype=float)[codes]
table = one_hot.sum(axis=1)

positions = np.arange(q, dtype=float)
weights = 1 - ((positions[:, None] - positions[None, :]) / (q - 1)) ** 2

ratings = table.sum(axis=1)
pa = (
    (
        np.einsum("ik,kl,il->i", table, weights, table)
        - ratings
    )
    / (ratings * (ratings - 1))
).mean()
proportions = table.sum(axis=0) / table.sum()
pe = float(proportions @ weights @ proportions)
kappa = float((pa - pe) / (1 - pe))

rng = np.random.default_rng(GRAINE)
bootstrap = []

for _ in range(N_BOOTSTRAP):
    indices = rng.integers(0, len(data), len(data))
    sampled_table = one_hot[indices].sum(axis=1)
    sampled_ratings = sampled_table.sum(axis=1)
    sampled_pa = (
        (
            np.einsum(
                "ik,kl,il->i",
                sampled_table,
                weights,
                sampled_table,
            )
            - sampled_ratings
        )
        / (sampled_ratings * (sampled_ratings - 1))
    ).mean()
    sampled_proportions = sampled_table.sum(axis=0) / sampled_table.sum()
    sampled_pe = float(sampled_proportions @ weights @ sampled_proportions)
    sampled_kappa = (sampled_pa - sampled_pe) / (1 - sampled_pe)

    if not np.isnan(sampled_kappa):
        bootstrap.append(sampled_kappa)

ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

summary = {
    "n_patients": int(len(pivot)),
    "annotators": ANNOTATEURS,
    "grades_observed": observed,
    "kappa_quadratic": kappa,
    "bootstrap_kappa_quadratic": {
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": GRAINE,
    },
}

(RESULTS / "calibration_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
