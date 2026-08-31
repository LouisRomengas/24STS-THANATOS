from pathlib import Path

import edsnlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, datediff, floor, length, lit, regexp_replace, when

DATASETS = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_datasets")
FINAL = DATASETS / "FINAL"
RESULTS = Path("/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_results/table1")
MAPPING = DATASETS / "correspondance_ids_20250605_121018.csv"
PARQUET = "hdfs:///user/cse240018/donnees_chirurgies_comptes_rendus"
DB = "cse240018_20240704_140053710028"

RESULTS.mkdir(parents=True, exist_ok=True)

spark.sql(f"USE {DB}")

full = spark.read.parquet(PARQUET)

person = spark.sql(
    """
    SELECT person_id, gender_source_value, birth_datetime, death_datetime
    FROM person
    """
)

care_site = spark.sql(
    """
    SELECT v.visit_occurrence_id, cs.care_site_name
    FROM visit_occurrence v
    LEFT JOIN care_site cs ON v.care_site_id = cs.care_site_id
    """
)

full = (
    full.join(person, on="person_id", how="left")
    .join(care_site, on="visit_occurrence_id", how="left")
    .withColumn(
        "age",
        floor(datediff(col("major_surgery_date"), col("birth_datetime")) / 365.25),
    )
    .withColumn(
        "dead",
        col("death_datetime").isNotNull()
        & (datediff(col("death_datetime"), col("major_surgery_date")) >= 0)
        & (datediff(col("death_datetime"), col("major_surgery_date")) <= 90),
    )
    .withColumn(
        "n_cr",
        (
            (
                length(col("cr_info"))
                - length(regexp_replace(col("cr_info"), '"date":', ""))
            )
            / lit(len('"date":'))
        ).cast("int"),
    )
    .withColumn(
        "n_ccam",
        (
            (
                length(col("followup_info"))
                - length(regexp_replace(col("followup_info"), '"ccam":', ""))
            )
            / lit(len('"ccam":'))
        ).cast("int"),
    )
    .withColumn(
        "n_cim10_before",
        (
            (
                length(col("diagnoses_info"))
                - length(regexp_replace(col("diagnoses_info"), '"days_before":', ""))
            )
            / lit(len('"days_before":'))
        ).cast("int"),
    )
    .withColumn(
        "_n_after_total",
        (
            (
                length(col("diagnoses_info"))
                - length(regexp_replace(col("diagnoses_info"), '"days_after":', ""))
            )
            / lit(len('"days_after":'))
        ).cast("int"),
    )
    .withColumn(
        "_diag_marked",
        regexp_replace(
            col("diagnoses_info"),
            r'"days_after":\s*(9[1-9]|[1-9][0-9][0-9])\b',
            "@@@",
        ),
    )
    .withColumn(
        "_n_after_over90",
        (
            (
                length(col("_diag_marked"))
                - length(regexp_replace(col("_diag_marked"), "@@@", ""))
            )
            / lit(3)
        ).cast("int"),
    )
    .withColumn("n_cim10_after", col("_n_after_total") - col("_n_after_over90"))
    .drop("_n_after_total", "_diag_marked", "_n_after_over90")
)

for column in ["n_cr", "n_ccam", "n_cim10_before", "n_cim10_after"]:
    full = full.withColumn(
        column,
        when(col(column).isNull() | (col(column) < 0), 0).otherwise(col(column)),
    )

full = (
    full.select(
        col("person_id").cast("string").alias("person_id"),
        "major_surgery_date",
        "age",
        "gender_source_value",
        "dead",
        "n_cr",
        "n_ccam",
        "n_cim10_before",
        "n_cim10_after",
        "care_site_name",
    )
    .toPandas()
)

full["major_surgery_date"] = pd.to_datetime(full["major_surgery_date"], errors="coerce")
full["date_chirurgie_str"] = full["major_surgery_date"].dt.strftime("%Y-%m-%d")
full = full.drop_duplicates(["person_id", "date_chirurgie_str"], keep="first")

train = pd.read_csv(FINAL / "train_cd1to5.csv", low_memory=False)
validation = pd.read_csv(FINAL / "val_cd1to5.csv", low_memory=False)
train["split"] = "train"
validation["split"] = "val"
reports = pd.concat([train, validation], ignore_index=True)

reports["patient_id"] = reports["patient_id"].fillna("").astype(str).str.strip()
reports["cd_manuel"] = pd.to_numeric(reports["cd_manuel"], errors="coerce")
reports.loc[reports["cd_manuel"] == 0, "cd_manuel"] = 1
reports = reports.loc[reports["cd_manuel"].between(1, 5)].copy()

mapping = pd.read_csv(MAPPING, dtype={"id_anonymise": str, "id_reel": str})
mapping = mapping[["id_anonymise", "id_reel", "date_chirurgie"]].drop_duplicates()
mapping.columns = ["patient_id", "person_id", "date_chirurgie"]
mapping["patient_id"] = mapping["patient_id"].fillna("").astype(str).str.strip()
mapping["person_id"] = mapping["person_id"].astype(str)
mapping["date_chirurgie_str"] = pd.to_datetime(
    mapping["date_chirurgie"],
    errors="coerce",
).dt.strftime("%Y-%m-%d")

patients = reports.drop_duplicates(["patient_id", "date_chirurgie"]).copy()
patients["date_chirurgie_str"] = pd.to_datetime(
    patients["date_chirurgie"],
    errors="coerce",
).dt.strftime("%Y-%m-%d")

patients = patients.merge(
    mapping[["patient_id", "person_id", "date_chirurgie_str"]],
    on=["patient_id", "date_chirurgie_str"],
    how="left",
)

patients = patients.merge(
    full[
        [
            "person_id",
            "date_chirurgie_str",
            "age",
            "gender_source_value",
            "dead",
            "n_cr",
            "n_ccam",
            "n_cim10_before",
            "n_cim10_after",
            "care_site_name",
        ]
    ],
    on=["person_id", "date_chirurgie_str"],
    how="left",
)

patients = patients.drop_duplicates("patient_id", keep="first").copy()

patients["category"] = (
    patients["categorie_chirurgie"]
    .astype(str)
    .str.strip()
    .map(
        {
            "oesophage_estomac": "Esophagus/Stomach",
            "colon": "Colon",
            "rectum": "Rectum",
            "foie_pancreas": "HPB",
            "voies_biliaires": "HPB",
            "foie_pancreas_voies_biliaires": "HPB",
            "autre": "Other",
        }
    )
    .fillna("Other")
)

patients["year"] = pd.to_datetime(
    patients["date_chirurgie"],
    errors="coerce",
).dt.year

patient_grades = (
    reports.groupby("patient_id", as_index=False)["cd_manuel"]
    .max()
    .rename(columns={"cd_manuel": "cd_max_patient"})
)

patients = patients.merge(patient_grades, on="patient_id", how="left")

nlp = edsnlp.blank("eds")

for variant, column in {"raw": "cr_text_brut", "cleaned": "cr_text_cleaned"}.items():
    texts = reports[column].fillna("").astype(str)
    cache = {text: len(nlp(text)) for text in texts.drop_duplicates()}
    reports[f"n_tokens_{variant}"] = texts.map(cache).astype(int)

patient_tokens = (
    reports.groupby(["patient_id", "split"], as_index=False)
    .agg(
        n_tokens_raw=("n_tokens_raw", "sum"),
        n_tokens_cleaned=("n_tokens_cleaned", "sum"),
    )
)

patients = patients.merge(patient_tokens, on=["patient_id", "split"], how="left")

groups = {
    "Overall": patients,
    "Train": patients.loc[patients["split"] == "train"],
    "Val": patients.loc[patients["split"] == "val"],
}

stats = {}

for name, frame in groups.items():
    n = len(frame)
    stats[("Patients, n", name)] = str(n)

    q1, median, q3 = np.percentile(
        pd.to_numeric(frame["age"], errors="coerce").dropna(),
        [25, 50, 75],
    )
    stats[("Age, median [IQR]", name)] = (
        f"{int(round(median))} [{int(round(q1))}-{int(round(q3))}]"
    )

    count = int(frame["gender_source_value"].astype(str).str.lower().eq("f").sum())
    stats[("Female, n (%)", name)] = f"{count} ({100 * count / n:.1f})"

    count = int(frame["dead"].fillna(False).astype(bool).sum())
    stats[("Death within 90-day follow-up, n (%)", name)] = (
        f"{count} ({100 * count / n:.1f})"
    )

    for category in ["Esophagus/Stomach", "Colon", "Rectum", "HPB", "Other"]:
        count = int(frame["category"].eq(category).sum())
        stats[(f"Category: {category}, n (%)", name)] = (
            f"{count} ({100 * count / n:.1f})"
        )

    for year in sorted(frame["year"].dropna().unique()):
        count = int(frame["year"].eq(year).sum())
        stats[(f"Surgery year {int(year)}, n (%)", name)] = (
            f"{count} ({100 * count / n:.1f})"
        )

    for label, column in [
        ("Index-surgery reports, median [IQR]", "n_cr"),
        ("Postoperative CCAM procedures (J+1 to J+90), median [IQR]", "n_ccam"),
        ("Prior ICD-10 codes (1 year before surgery), median [IQR]", "n_cim10_before"),
        ("Postoperative ICD-10 codes (J+1 to J+90), median [IQR]", "n_cim10_after"),
    ]:
        q1, median, q3 = np.percentile(
            pd.to_numeric(frame[column], errors="coerce").dropna(),
            [25, 50, 75],
        )
        stats[(label, name)] = (
            f"{int(round(median))} [{int(round(q1))}-{int(round(q3))}]"
        )

    count = int(frame["n_ccam"].eq(0).sum())
    stats[("No postop CCAM, n (%)", name)] = f"{count} ({100 * count / n:.1f})"

    stats[("Hospital sites, n", name)] = str(
        frame["care_site_name"].dropna().nunique()
    )

    for grade, roman in zip([1, 2, 3, 4, 5], ["I", "II", "III", "IV", "V"]):
        count = int(frame["cd_max_patient"].eq(grade).sum())
        stats[(f"CD-{roman}, n (%)", name)] = f"{count} ({100 * count / n:.1f})"

    count = int(frame["cd_max_patient"].ge(3).sum())
    stats[("CD-III or higher (severe), n (%)", name)] = (
        f"{count} ({100 * count / n:.1f})"
    )

    stats[("Annotated reports, n", name)] = str(
        len(reports)
        if name == "Overall"
        else len(reports.loc[reports["split"] == name.lower()])
    )

    for variant in ["raw", "cleaned"]:
        q1, median, q3 = np.percentile(
            frame[f"n_tokens_{variant}"].dropna(),
            [25, 50, 75],
        )
        stats[(f"Report length ({variant} text), tokens", name)] = (
            f"{int(round(median))} [{int(round(q1))}-{int(round(q3))}]"
        )

variables = [
    "Patients, n",
    "Age, median [IQR]",
    "Female, n (%)",
    "Death within 90-day follow-up, n (%)",
    "Category: Esophagus/Stomach, n (%)",
    "Category: Colon, n (%)",
    "Category: Rectum, n (%)",
    "Category: HPB, n (%)",
    "Category: Other, n (%)",
]

for year in sorted(patients["year"].dropna().unique()):
    variables.append(f"Surgery year {int(year)}, n (%)")

variables += [
    "Index-surgery reports, median [IQR]",
    "Postoperative CCAM procedures (J+1 to J+90), median [IQR]",
    "No postop CCAM, n (%)",
    "Prior ICD-10 codes (1 year before surgery), median [IQR]",
    "Postoperative ICD-10 codes (J+1 to J+90), median [IQR]",
    "Hospital sites, n",
    "CD-I, n (%)",
    "CD-II, n (%)",
    "CD-III, n (%)",
    "CD-IV, n (%)",
    "CD-V, n (%)",
    "CD-III or higher (severe), n (%)",
    "Annotated reports, n",
    "Report length (raw text), tokens",
    "Report length (cleaned text), tokens",
]

table = pd.DataFrame(
    [
        [
            variable,
            stats[(variable, "Overall")],
            stats[(variable, "Train")],
            stats[(variable, "Val")],
        ]
        for variable in variables
    ],
    columns=[
        "Variable",
        f"Overall (n={len(groups['Overall'])})",
        f"Train (n={len(groups['Train'])})",
        f"Val (n={len(groups['Val'])})",
    ],
)

table.to_csv(
    RESULTS / "Table1-programmatic-components.csv",
    index=False,
    encoding="utf-8-sig",
)

x = np.arange(5)
width = 0.25
figure, axes = plt.subplots(2, 1, figsize=(8, 10))

for axis, level, panel in [
    (axes[0], "patient", "A"),
    (axes[1], "report", "B"),
]:
    for offset, label, frame in [
        (-width, "Training", train),
        (0, "Validation", validation),
        (width, "Overall", reports),
    ]:
        grades = (
            frame.groupby("patient_id")["cd_manuel"].max()
            if level == "patient"
            else frame["cd_manuel"]
        )
        counts = grades.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        axis.bar(
            x + offset,
            100 * counts.to_numpy() / counts.sum(),
            width,
            label=label,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(["CD-I", "CD-II", "CD-III", "CD-IV", "CD-V"])
    axis.set_xlabel("Clavien-Dindo grade")
    axis.set_ylabel("Percentage (%)")
    axis.legend(title="Dataset")
    axis.text(0.01, 0.96, panel, transform=axis.transAxes, va="top")

figure.tight_layout()
figure.savefig(RESULTS / "FigureS4.pdf", bbox_inches="tight")
figure.savefig(RESULTS / "FigureS4.png", dpi=300, bbox_inches="tight")
plt.close(figure)
