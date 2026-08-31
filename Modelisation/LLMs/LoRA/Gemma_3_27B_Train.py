import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from prompts import SYSTEM_PROMPT, KEYS, DIRECT_PROMPT_TEMPLATE, TWO_STEP_PROMPT_TEMPLATE

DATA_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/FINAL"
)

INTERNAL_TRAIN = DATA_DIR / "lora_train-set_cd1to5_internal_train.csv"
INTERNAL_VAL = DATA_DIR / "lora_train-set_cd1to5_internal_validation.csv"
FULL_TRAIN = DATA_DIR / "lora_train-set_cd1to5.csv"

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-27b-it")
EXPERIMENT = "gemma3_27b_qlora"
PAD_ID = 0


def encode(tokenizer, user_prompt, assistant=None, eos_id=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=assistant is None,
        tokenize=True,
        return_dict=False,
    )

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    if ids and isinstance(ids[0], list):
        ids = ids[0]

    ids = [int(x) for x in ids]

    if assistant is not None and ids[-1] != eos_id:
        ids.append(int(eos_id))

    return ids


def make_dataset(df, tokenizer, text_mode, target_mode, max_prompt_tokens, max_seq_length, eos_id):
    text_col = "cr_text_cleaned" if text_mode == "cleaned" else "cr_text_brut"
    template = TWO_STEP_PROMPT_TEMPLATE if target_mode == "twostep" else DIRECT_PROMPT_TEMPLATE
    samples = []

    required = {
        "patient_id",
        "cd_manuel",
        "cr_text_brut",
        "cr_text_cleaned",
        "json_manuel",
        "major_surgery_label",
        "date_chirurgie",
        "date_cr",
    }

    assert required.issubset(df.columns)

    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        grade = int(row["cd_manuel"])

        assert text
        assert 1 <= grade <= 5

        if target_mode == "twostep":
            source = json.loads(str(row["json_manuel"]))
            assert set(source) == set(KEYS)
            labels = {}

            for key in KEYS:
                value = str(source[key]).strip().lower()
                assert value in {"oui", "non", "yes", "no", "true", "false", "1", "0"}
                labels[key] = "oui" if value in {"oui", "yes", "true", "1"} else "non"

            completion = json.dumps(
                labels,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            completion = str(grade)

        context = {
            "date_chirurgie": str(row["date_chirurgie"]),
            "surgery_label": str(row["major_surgery_label"]),
            "date_cr": str(row["date_cr"]),
        }

        sentences = [
            x.strip()
            for x in re.split(r"(?<=[\.\!\?;:])\s+|\n+", text)
            if x.strip()
        ]

        chunks = []
        current = []

        for sentence in sentences:
            candidate = " ".join(current + [sentence]).strip()
            prompt = template.format(cr_text=candidate, **context)

            if len(encode(tokenizer, prompt)) <= max_prompt_tokens:
                current.append(sentence)
            elif current:
                chunks.append(" ".join(current).strip())
                current = [sentence]
            else:
                chunks.append(sentence)

        if current:
            chunks.append(" ".join(current).strip())

        if not chunks:
            chunks = [text]

        for chunk in chunks:
            prompt = template.format(cr_text=chunk, **context)
            prompt_ids = encode(tokenizer, prompt)
            full_ids = encode(tokenizer, prompt, completion, eos_id)
            prefix = 0

            while (
                prefix < len(prompt_ids)
                and prefix < len(full_ids)
                and prompt_ids[prefix] == full_ids[prefix]
            ):
                prefix += 1

            assert 0 < prefix < len(full_ids)

            if len(full_ids) <= max_seq_length:
                samples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": [-100] * prefix + full_ids[prefix:],
                    }
                )

    assert samples
    return Dataset.from_list(samples)


def collate(features):
    max_len = ((max(len(x["input_ids"]) for x in features) + 7) // 8) * 8

    return {
        "input_ids": torch.tensor(
            [
                x["input_ids"] + [PAD_ID] * (max_len - len(x["input_ids"]))
                for x in features
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                x["attention_mask"] + [0] * (max_len - len(x["attention_mask"]))
                for x in features
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [
                x["labels"] + [-100] * (max_len - len(x["labels"]))
                for x in features
            ],
            dtype=torch.long,
        ),
    }


def load_model(token):
    source = os.environ.get("MODEL_PATH", "").strip() or MODEL_ID
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        source,
        token=token,
        quantization_config=quantization,
        device_map="auto",
        attn_implementation="eager",
    )

    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    assert "exclude_modules" in inspect.signature(LoraConfig.__init__).parameters

    return get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            exclude_modules=r".*(vision_tower|vision_model|multi_modal_projector|mm_projector).*",
        ),
    )


def run(args):
    global PAD_ID

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source = os.environ.get("MODEL_PATH", "").strip() or MODEL_ID
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        token=token,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 8192
    PAD_ID = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)

    if args.run_mode == "development":
        train_path = Path(args.train_csv or INTERNAL_TRAIN)
        val_path = Path(args.internal_val_csv or INTERNAL_VAL)
        epochs = float(args.num_train_epochs)
    else:
        train_path = Path(args.train_csv or FULL_TRAIN)
        val_path = None
        selection = json.loads(Path(args.selection_meta_json).read_text(encoding="utf-8"))
        assert selection.get("training_role") == "development_model_selection"
        selected_epoch = float(selection["best_epoch"])
        rounded_epoch = int(round(selected_epoch))
        assert rounded_epoch >= 1
        assert np.isclose(selected_epoch, rounded_epoch, atol=1e-6)
        epochs = float(rounded_epoch)

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False) if val_path else None

    if args.run_mode == "development":
        assert len(train_df) == 224
        assert train_df["patient_id"].astype(str).nunique() == 140
        assert len(val_df) == 58
        assert val_df["patient_id"].astype(str).nunique() == 35
        assert not set(train_df["patient_id"].astype(str)) & set(val_df["patient_id"].astype(str))
    else:
        assert len(train_df) == 282
        assert train_df["patient_id"].astype(str).nunique() == 175

    train_ds = make_dataset(
        train_df,
        tokenizer,
        args.text_mode,
        args.target_mode,
        args.max_prompt_tokens,
        args.max_seq_length,
        eos_id,
    )

    val_ds = (
        make_dataset(
            val_df,
            tokenizer,
            args.text_mode,
            args.target_mode,
            args.max_prompt_tokens,
            args.max_seq_length,
            eos_id,
        )
        if val_df is not None
        else None
    )

    model = load_model(token)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_dir": str(output_dir / "trainer_outputs"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "report_to": "none",
        "logging_strategy": "no",
        "disable_tqdm": True,
        "optim": "paged_adamw_8bit",
        "max_grad_norm": 1.0,
        "seed": args.seed,
        "data_seed": args.seed,
    }

    if args.run_mode == "development":
        kwargs.update(
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            kwargs["eval_strategy"] = "epoch"
        else:
            kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs.update(
            save_strategy="no",
            load_best_model_at_end=False,
        )

        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            kwargs["eval_strategy"] = "no"
        else:
            kwargs["evaluation_strategy"] = "no"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )

    tracker = OfflineEmissionsTracker(
        project_name="THANATOS",
        experiment_id=f"{EXPERIMENT}_{args.text_mode}_{args.target_mode}_{args.run_mode}",
        output_dir=str(output_dir),
        output_file="emissions.csv",
        log_level="critical",
        tracking_mode="process",
        country_iso_code="FRA",
    )

    tracker.start()
    start = time.perf_counter()
    result = trainer.train()
    tracker.stop()
    wall_seconds = time.perf_counter() - start

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    best_epoch = None
    best_eval_loss = None

    if args.run_mode == "development":
        records = [
            x
            for x in trainer.state.log_history
            if x.get("eval_loss") is not None
        ]
        assert records
        best = min(records, key=lambda x: float(x["eval_loss"]))
        best_epoch = float(best["epoch"])
        best_eval_loss = float(best["eval_loss"])

    metadata = {
        "model_id": MODEL_ID,
        "method": "qlora",
        "quantization": "nf4_double_quantization",
        "compute_dtype": "bfloat16",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear_language_decoder",
        "learning_rate": args.learning_rate,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "optimizer": "paged_adamw_8bit",
        "max_grad_norm": 1.0,
        "effective_batch_size": 16,
        "loss": "completion_only",
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_seq_length": args.max_seq_length,
        "run_mode": args.run_mode,
        "training_role": "development_model_selection"
        if args.run_mode == "development"
        else "final_refit",
        "text_mode": args.text_mode,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "train_csv": str(train_path.resolve()),
        "internal_validation_csv": str(val_path.resolve()) if val_path else None,
        "external_test_used": False,
        "train_reports": int(len(train_df)),
        "train_patients": int(train_df["patient_id"].nunique()),
        "internal_validation_reports": int(len(val_df)) if val_df is not None else None,
        "internal_validation_patients": int(val_df["patient_id"].nunique()) if val_df is not None else None,
        "num_train_epochs": epochs,
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "training_wall_seconds": float(wall_seconds),
        "trainer_metrics": result.metrics,
        "adapter_dir": str(adapter_dir.resolve()),
    }

    (output_dir / "training_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


parser = argparse.ArgumentParser()
parser.add_argument("--run-mode", choices=["development", "final_refit"], default="development")
parser.add_argument("--train-csv")
parser.add_argument("--internal-val-csv")
parser.add_argument("--selection-meta-json")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--text-mode", choices=["raw", "cleaned"], required=True)
parser.add_argument("--target-mode", choices=["direct", "twostep"], required=True)
parser.add_argument("--max-prompt-tokens", type=int, default=5800)
parser.add_argument("--max-seq-length", type=int, default=6144)
parser.add_argument("--num-train-epochs", type=float, default=3.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

run(args)
