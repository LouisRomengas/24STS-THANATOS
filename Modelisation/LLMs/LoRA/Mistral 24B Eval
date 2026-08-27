import argparse
import itertools
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from transformers import BitsAndBytesConfig, Mistral3ForConditionalGeneration

from prompts import SYSTEM_PROMPT, KEYS, DIRECT_PROMPT_TEMPLATE, TWO_STEP_PROMPT_TEMPLATE

DATA_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/FINAL"
)

FULL_TRAIN = DATA_DIR / "lora_train-set_cd1to5.csv"
EXTERNAL_TEST = DATA_DIR / "lora_val_cd1to5.csv"

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
)

LABELS = [1, 2, 3, 4, 5]


def encode(tokenizer, user_prompt, assistant=None, eos_id=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    request = ChatCompletionRequest(
        messages=messages
        if assistant is None
        else messages + [{"role": "assistant", "content": assistant}],
        continue_final_message=assistant is not None,
    )

    ids = [int(x) for x in tokenizer.encode_chat_completion(request).tokens]

    if assistant is not None and ids[-1] != eos_id:
        ids.append(int(eos_id))

    return ids


def labels_to_grade(labels):
    yes = lambda key: str(labels[key]).strip().lower() == "oui"

    if yes("deces_lie_complication") and yes("evenement_temporel_pertinent"):
        return 5

    if (
        yes("complication_nouvelle")
        and yes("evenement_temporel_pertinent")
        and yes("usc_rea_non_programmee_pour_complication")
        and yes("defaillance_organe")
    ):
        return 4

    if (
        yes("complication_nouvelle")
        and yes("evenement_temporel_pertinent")
        and yes("geste_invasif_therapeutique")
    ):
        return 3

    if (
        yes("complication_nouvelle")
        and yes("evenement_temporel_pertinent")
        and yes("traitement_pharmacologique_specifique")
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


def load_model(adapter_dir, token):
    source = os.environ.get("MODEL_PATH", "").strip() or MODEL_ID
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Mistral3ForConditionalGeneration.from_pretrained(
        source,
        token=token,
        quantization_config=quantization,
        device_map="auto",
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.config.use_cache = True

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True

    return model


def run(args):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source = os.environ.get("MODEL_PATH", "").strip() or MODEL_ID
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = MistralTokenizer.from_hf_hub(source)
    inner = tokenizer.instruct_tokenizer.tokenizer
    eos_id = int(inner.eos_id)
    pad_id = int(inner.pad_id) if isinstance(inner.pad_id, int) and inner.pad_id >= 0 else eos_id

    test_path = Path(args.external_test_csv or EXTERNAL_TEST)
    train_path = Path(args.full_train_csv or FULL_TRAIN)
    adapter_dir = Path(args.adapter_dir)

    test = pd.read_csv(test_path, low_memory=False)
    train = pd.read_csv(train_path, usecols=["patient_id"], low_memory=False)

    test["patient_id"] = test["patient_id"].fillna("").astype(str).str.strip()
    train["patient_id"] = train["patient_id"].fillna("").astype(str).str.strip()

    assert not set(test["patient_id"]) & set(train["patient_id"])

    meta_path = adapter_dir.parent / "training_meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("training_role") == "final_refit"

    text_col = "cr_text_cleaned" if args.text_mode == "cleaned" else "cr_text_brut"
    required = {
        "patient_id",
        "cd_manuel",
        text_col,
        "major_surgery_label",
        "date_chirurgie",
        "date_cr",
    }

    assert required.issubset(test.columns)

    test[text_col] = test[text_col].fillna("").astype(str)
    test = test[test[text_col].str.strip().ne("")].copy()
    test["cd_manuel"] = pd.to_numeric(test["cd_manuel"], errors="coerce")
    test = test[test["cd_manuel"].between(1, 5)].copy()
    test["cd_manuel"] = test["cd_manuel"].astype(int)

    model = load_model(adapter_dir, token)
    template = TWO_STEP_PROMPT_TEMPLATE if args.target_mode == "twostep" else DIRECT_PROMPT_TEMPLATE

    valid_outputs = (
        [
            json.dumps(dict(zip(KEYS, values)), ensure_ascii=False)
            for values in itertools.product(("non", "oui"), repeat=len(KEYS))
        ]
        if args.target_mode == "twostep"
        else [str(x) for x in LABELS]
    )

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        project_name="THANATOS",
        experiment_id=f"mistral24b_qlora_{args.text_mode}_{args.target_mode}_evaluation",
        output_dir=str(output_dir),
        output_file="emissions.csv",
        log_level="critical",
        tracking_mode="process",
        country_iso_code="FRA",
    )

    tracker.start()
    start = time.perf_counter()
    rows = []

    for _, row in test.iterrows():
        context = {
            "date_chirurgie": str(row["date_chirurgie"]),
            "surgery_label": str(row["major_surgery_label"]),
            "date_cr": str(row["date_cr"]),
        }

        sentences = [
            x.strip()
            for x in re.split(r"(?<=[\.\!\?;:])\s+|\n+", str(row[text_col]).strip())
            if x.strip()
        ]

        chunks = []
        current = []

        for sentence in sentences:
            candidate = " ".join(current + [sentence]).strip()
            prompt = template.format(cr_text=candidate, **context)

            if len(encode(tokenizer, prompt)) <= args.max_input_tokens:
                current.append(sentence)
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = [sentence]
                else:
                    chunks.append(sentence)

        if current:
            chunks.append(" ".join(current).strip())

        if not chunks:
            chunks = [str(row[text_col]).strip()]

        chunk_results = []

        for chunk in chunks:
            prompt = template.format(cr_text=chunk, **context)
            prompt_ids = encode(tokenizer, prompt)
            candidates = {}

            for text in valid_outputs:
                full_ids = encode(tokenizer, prompt, text, eos_id)
                candidates[tuple(full_ids[len(prompt_ids):])] = text

            trie = {}

            for sequence in candidates:
                node = trie

                for token_id in sequence:
                    node = node.setdefault(int(token_id), {})

            prompt_len = len(prompt_ids)

            def allowed_tokens(_, sent):
                generated = [int(x) for x in sent.tolist()[prompt_len:]]
                node = trie

                for token_id in generated:
                    node = node[int(token_id)]

                return [int(x) for x in node.keys()] or [eos_id]

            inputs = {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=model.device),
                "attention_mask": torch.ones(
                    (1, len(prompt_ids)),
                    dtype=torch.long,
                    device=model.device,
                ),
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max(len(x) for x in candidates) + 1,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    prefix_allowed_tokens_fn=allowed_tokens,
                )[0][prompt_len:].tolist()

            sequence = tuple(int(x) for x in generated_ids)

            if sequence not in candidates and sequence + (eos_id,) in candidates:
                sequence = sequence + (eos_id,)

            generated = candidates[sequence]

            if args.target_mode == "twostep":
                labels = json.loads(generated)
                grade = labels_to_grade(labels)
            else:
                labels = None
                grade = int(generated)

            chunk_results.append(
                {
                    "grade": int(grade),
                    "output": generated,
                    "labels": labels,
                }
            )

        selected = max(chunk_results, key=lambda x: x["grade"])
        output = row.to_dict()
        output["cd_manuel"] = int(row["cd_manuel"])
        output["cd_pred"] = int(selected["grade"])
        output["n_chunks"] = len(chunk_results)
        output["llm_output"] = selected["output"]

        if args.target_mode == "twostep":
            output["pred_labels"] = json.dumps(selected["labels"], ensure_ascii=False)

        rows.append(output)

    tracker.stop()
    wall_seconds = time.perf_counter() - start

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "val_predictions_cr_level.csv", index=False)

    report_metrics = scores(report["cd_manuel"], report["cd_pred"])
    report_metrics["confusion_matrix"] = confusion_matrix(
        report["cd_manuel"],
        report["cd_pred"],
        labels=LABELS,
    ).tolist()
    report_metrics["classification_report"] = classification_report(
        report["cd_manuel"],
        report["cd_pred"],
        labels=LABELS,
        target_names=["CD-I", "CD-II", "CD-III", "CD-IV", "CD-V"],
        output_dict=True,
        zero_division=0,
    )

    patient = (
        report.groupby("patient_id", as_index=False)
        .agg(
            cd_manuel=("cd_manuel", "max"),
            cd_pred=("cd_pred", "max"),
        )
    )

    patient.to_csv(output_dir / "val_predictions_patient_level.csv", index=False)

    patient_metrics = scores(patient["cd_manuel"], patient["cd_pred"])
    patient_metrics["confusion_matrix"] = confusion_matrix(
        patient["cd_manuel"],
        patient["cd_pred"],
        labels=LABELS,
    ).tolist()
    patient_metrics["classification_report"] = classification_report(
        patient["cd_manuel"],
        patient["cd_pred"],
        labels=LABELS,
        target_names=["CD-I", "CD-II", "CD-III", "CD-IV", "CD-V"],
        output_dict=True,
        zero_division=0,
    )

    bootstrap_report = bootstrap(
        report["cd_manuel"],
        report["cd_pred"],
        args.n_bootstrap_val,
        args.seed,
    )

    bootstrap_patient = bootstrap(
        patient["cd_manuel"],
        patient["cd_pred"],
        args.n_bootstrap_val,
        args.seed,
    )

    (output_dir / "val_results_cr_level.json").write_text(
        json.dumps(report_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "val_results_patient_level.json").write_text(
        json.dumps(patient_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "bootstrap_val_cr_level.json").write_text(
        json.dumps(bootstrap_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "bootstrap_val_patient_level.json").write_text(
        json.dumps(bootstrap_patient, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "model_id": MODEL_ID,
        "method": "qlora",
        "adapter_dir": str(adapter_dir.resolve()),
        "text_mode": args.text_mode,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "dataset_role": "external_held_out_test",
        "external_test_csv": str(test_path.resolve()),
        "full_train_csv": str(train_path.resolve()),
        "n_test_cr": int(len(report)),
        "n_test_patients": int(patient["patient_id"].nunique()),
        "runtime_seconds": float(wall_seconds),
        "val_results_cr_level": report_metrics,
        "val_results_patient_level": patient_metrics,
        "bootstrap_val_cr_level": bootstrap_report,
        "bootstrap_val_patient_level": bootstrap_patient,
    }

    (output_dir / "paper_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


parser = argparse.ArgumentParser()
parser.add_argument("--external-test-csv")
parser.add_argument("--full-train-csv")
parser.add_argument("--results-dir", required=True)
parser.add_argument("--adapter-dir", required=True)
parser.add_argument("--text-mode", choices=["raw", "cleaned"], required=True)
parser.add_argument("--target-mode", choices=["direct", "twostep"], required=True)
parser.add_argument("--max-input-tokens", type=int, default=5800)
parser.add_argument("--n-bootstrap-val", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

run(args)
