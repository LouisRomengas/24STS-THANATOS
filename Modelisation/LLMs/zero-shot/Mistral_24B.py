import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from prompts import SYSTEM_PROMPT, KEYS, DIRECT_PROMPT_TEMPLATE, TWO_STEP_PROMPT_TEMPLATE

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)

DATA_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/FINAL"
)

TRAIN_PATH = DATA_DIR / "train_cd1to5.csv"
VAL_PATH = DATA_DIR / "val_cd1to5.csv"

CCAM_DICT_PATH = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/ccam_surgery_dictionary.csv"
)

OUTPUT_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_results/llm_mistral24b_bf16_cd1to5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
)

LABELS = [1, 2, 3, 4, 5]
LABEL_NAMES = ["CD-I", "CD-II", "CD-III", "CD-IV", "CD-V"]
CANDIDATES = ["1", "2", "3", "4", "5"]
COL_MAP = {
    "cleaned": "cr_text_cleaned",
    "raw": "cr_text_brut",
}


class TwoStepExtraction(BaseModel):
    complication_nouvelle: bool
    evenement_temporel_pertinent: bool
    traitement_pharmacologique_specifique: bool
    geste_invasif_therapeutique: bool
    usc_rea_surveillance_programmee: bool
    usc_rea_non_programmee_pour_complication: bool
    defaillance_organe: bool
    deces_lie_complication: bool


def render_prompt(mode, row, cr_text):
    template = TWO_STEP_PROMPT_TEMPLATE if mode == "twostep" else DIRECT_PROMPT_TEMPLATE
    return template.format(
        date_chirurgie=row["date_chirurgie"],
        surgery_label=row["surgery_label"],
        date_cr=row["date_cr"],
        cr_text=cr_text,
    )


def build_chunks(text, tokenizer, row, mode, max_input_tokens, max_sentences):
    sentences = [
        x.strip()
        for x in re.split(r"(?<=[\.\!\?])\s+|\n+", str(text).strip())
        if x.strip()
    ]

    chunks = []
    current = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        prompt = SYSTEM_PROMPT + "\n\n" + render_prompt(mode, row, candidate)
        encoded = tokenizer.encode(prompt)
        token_ids = encoded if isinstance(encoded, list) else encoded.ids

        if len(token_ids) <= max_input_tokens and len(current) < max_sentences:
            current.append(sentence)
        else:
            if current:
                chunks.append(" ".join(current).strip())
                current = [sentence]
            else:
                chunks.append(sentence)

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def labels_to_grade(labels):
    if labels["deces_lie_complication"] and labels["evenement_temporel_pertinent"]:
        return 5

    if (
        labels["complication_nouvelle"]
        and labels["evenement_temporel_pertinent"]
        and labels["usc_rea_non_programmee_pour_complication"]
        and labels["defaillance_organe"]
    ):
        return 4

    if (
        labels["complication_nouvelle"]
        and labels["evenement_temporel_pertinent"]
        and labels["geste_invasif_therapeutique"]
    ):
        return 3

    if (
        labels["complication_nouvelle"]
        and labels["evenement_temporel_pertinent"]
        and labels["traitement_pharmacologique_specifique"]
    ):
        return 2

    return 1


def scores(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    severe_true = (y_true >= 3).astype(int)
    severe_pred = (y_pred >= 3).astype(int)
    tp = int(((severe_true == 1) & (severe_pred == 1)).sum())
    fn = int(((severe_true == 1) & (severe_pred == 0)).sum())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "severe_sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def bootstrap(y_true, y_pred, n_bootstrap, seed):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in scores(y_true, y_pred)}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample = scores(y_true[idx], y_pred[idx])

        for key in values:
            values[key].append(sample[key])

    return {
        key: {
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v, ddof=1)),
            "ci95_low": float(np.nanpercentile(v, 2.5)),
            "ci95_high": float(np.nanpercentile(v, 97.5)),
            "n_bootstrap": n_bootstrap,
        }
        for key, v in values.items()
    }


def run_config(
    llm,
    tokenizer,
    val,
    text_col,
    mode,
    max_input_tokens,
    max_sentences,
    n_bootstrap,
    seed,
    max_val_rows,
):
    df = val[
        [
            "patient_id",
            text_col,
            "cd_manuel",
            "date_chirurgie",
            "date_cr",
            "major_surgery_ccam",
            "surgery_label",
        ]
    ].copy()

    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip().ne("")].copy()
    df["cd_manuel"] = pd.to_numeric(df["cd_manuel"], errors="coerce")
    df = df[df["cd_manuel"].between(1, 5)].copy()
    df["cd_manuel"] = df["cd_manuel"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str)

    if max_val_rows > 0:
        df = df.iloc[:max_val_rows].copy()

    tag = "cleaned" if text_col == "cr_text_cleaned" else "raw"
    result_dir = OUTPUT_DIR / f"text_{tag}_mode_{mode}"
    result_dir.mkdir(parents=True, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        project_name="THANATOS",
        experiment_id=f"mistral24b_zeroshot_{tag}_{mode}",
        output_dir=str(result_dir),
        output_file=f"emissions_{tag}_{mode}.csv",
        log_level="critical",
        tracking_mode="process",
        country_iso_code="FRA",
    )

    tracker.start()
    start = time.perf_counter()

    predictions = []
    raw_outputs = []
    selected_labels = []

    schema = TwoStepExtraction.model_json_schema()

    for _, row in df.iterrows():
        chunks = build_chunks(
            row[text_col],
            tokenizer,
            row,
            mode,
            max_input_tokens,
            max_sentences,
        )

        grades = []
        outputs = []
        labels_per_chunk = []

        for chunk in chunks:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_prompt(mode, row, chunk)},
            ]

            if mode == "twostep":
                sampling = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=256,
                    seed=seed,
                    structured_outputs=StructuredOutputsParams(json=schema),
                )

                generated = llm.chat([messages], sampling)[0].outputs[0].text.strip()
                labels = TwoStepExtraction.model_validate_json(generated).model_dump()
                grade = labels_to_grade(labels)

                outputs.append(generated)
                labels_per_chunk.append(labels)

            else:
                sampling = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4,
                    seed=seed,
                    structured_outputs=StructuredOutputsParams(choice=CANDIDATES),
                )

                generated = llm.chat([messages], sampling)[0].outputs[0].text.strip()
                grade = int(generated)

                outputs.append(generated)
                labels_per_chunk.append(None)

            grades.append(int(grade))

        best = int(np.argmax(grades))
        predictions.append(grades[best])
        raw_outputs.append(outputs[best])
        selected_labels.append(labels_per_chunk[best])

    tracker.stop()
    wall_seconds = time.perf_counter() - start

    df["cd_pred"] = predictions
    df["llm_output"] = raw_outputs

    if mode == "twostep":
        for key in KEYS:
            df[f"signal_{key}"] = [
                labels[key] if labels is not None else False
                for labels in selected_labels
            ]

    df.to_csv(result_dir / "val_predictions_cr_level.csv", index=False)

    cr_results = scores(df["cd_manuel"], df["cd_pred"])
    cr_results["confusion_matrix"] = confusion_matrix(
        df["cd_manuel"],
        df["cd_pred"],
        labels=LABELS,
    ).tolist()
    cr_results["classification_report"] = classification_report(
        df["cd_manuel"],
        df["cd_pred"],
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    patient = (
        df.groupby("patient_id", as_index=False)
        .agg(
            cd_manuel=("cd_manuel", "max"),
            cd_pred=("cd_pred", "max"),
        )
    )

    patient.to_csv(result_dir / "val_predictions_patient_level.csv", index=False)

    patient_results = scores(patient["cd_manuel"], patient["cd_pred"])
    patient_results["confusion_matrix"] = confusion_matrix(
        patient["cd_manuel"],
        patient["cd_pred"],
        labels=LABELS,
    ).tolist()
    patient_results["classification_report"] = classification_report(
        patient["cd_manuel"],
        patient["cd_pred"],
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    bootstrap_cr = bootstrap(
        df["cd_manuel"],
        df["cd_pred"],
        n_bootstrap,
        seed,
    )

    bootstrap_patient = bootstrap(
        patient["cd_manuel"],
        patient["cd_pred"],
        n_bootstrap,
        seed,
    )

    (result_dir / "val_results_cr_level.json").write_text(
        json.dumps(cr_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (result_dir / "val_results_patient_level.json").write_text(
        json.dumps(patient_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (result_dir / "bootstrap_val_cr_level.json").write_text(
        json.dumps(bootstrap_cr, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (result_dir / "bootstrap_val_patient_level.json").write_text(
        json.dumps(bootstrap_patient, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    emissions_path = result_dir / f"emissions_{tag}_{mode}.csv"
    energy = {}

    if emissions_path.exists():
        emissions = pd.read_csv(emissions_path)

        if not emissions.empty:
            row = emissions.iloc[-1]
            energy = {
                "duration_seconds": float(row.get("duration", 0)),
                "energy_kwh": float(row.get("energy_consumed", 0)),
                "emissions_kg_co2eq": float(row.get("emissions", 0)),
            }

    paper_summary = {
        "model_id": MODEL_ID,
        "dataset_train": str(TRAIN_PATH),
        "dataset_test": str(VAL_PATH),
        "train_used_for_inference": False,
        "text_col": text_col,
        "mode": mode,
        "seed": seed,
        "n_val_reports": int(len(df)),
        "n_val_patients": int(df["patient_id"].nunique()),
        "wall_seconds": float(wall_seconds),
        "energy": energy,
        "val_results_cr_level": cr_results,
        "val_results_patient_level": patient_results,
        "bootstrap_val_cr_level": bootstrap_cr,
        "bootstrap_val_patient_level": bootstrap_patient,
    }

    (result_dir / "paper_summary.json").write_text(
        json.dumps(paper_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


parser = argparse.ArgumentParser()
parser.add_argument("--text_cols", nargs="+", default=["cleaned", "raw"], choices=["cleaned", "raw"])
parser.add_argument("--modes", nargs="+", default=["direct", "twostep"], choices=["direct", "twostep"])
parser.add_argument("--max_input_tokens", type=int, default=6000)
parser.add_argument("--max_sentences", type=int, default=80)
parser.add_argument("--n_bootstrap_val", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_val_rows", type=int, default=0)
parser.add_argument("--max_model_len", type=int, default=8192)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
parser.add_argument("--no_enforce_eager", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)

train_ids = set(
    pd.read_csv(TRAIN_PATH, usecols=["patient_id"])["patient_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

val = pd.read_csv(VAL_PATH, low_memory=False)
val["patient_id"] = val["patient_id"].fillna("").astype(str).str.strip()

assert not train_ids.intersection(set(val["patient_id"]))

ccam = pd.read_csv(CCAM_DICT_PATH)
ccam_dict = ccam.set_index("Code")["Libelle"].to_dict()

val["major_surgery_ccam"] = val["major_surgery_ccam"].fillna("N/A").astype(str)
val["surgery_label"] = val["major_surgery_ccam"].map(ccam_dict).fillna("Acte non trouvé")
val["date_chirurgie"] = val["date_chirurgie"].fillna("N/A").astype(str)
val["date_cr"] = val["date_cr"].fillna("N/A").astype(str)

llm = LLM(
    model=MODEL_ID,
    tokenizer_mode="mistral",
    config_format="mistral",
    load_format="mistral",
    dtype="bfloat16",
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True,
    enforce_eager=not args.no_enforce_eager,
    tensor_parallel_size=1,
    seed=args.seed,
    limit_mm_per_prompt={"image": 0},
)

tokenizer = llm.get_tokenizer()

for text_key in args.text_cols:
    for mode in args.modes:
        run_config(
            llm,
            tokenizer,
            val,
            COL_MAP[text_key],
            mode,
            args.max_input_tokens,
            args.max_sentences,
            args.n_bootstrap_val,
            args.seed,
            args.max_val_rows,
        )
