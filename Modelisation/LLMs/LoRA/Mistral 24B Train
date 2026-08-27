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
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Mistral3ForConditionalGeneration, Trainer, TrainingArguments
from transformers.utils import logging as hf_logging

from prompts import SYSTEM_PROMPT, KEYS, DIRECT_PROMPT_TEMPLATE, TWO_STEP_PROMPT_TEMPLATE

hf_logging.set_verbosity_error()

DATA_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/FINAL"
)

INTERNAL_TRAIN = DATA_DIR / "lora_train-set_cd1to5_internal_train.csv"
INTERNAL_VAL = DATA_DIR / "lora_train-set_cd1to5_internal_validation.csv"
FULL_TRAIN = DATA_DIR / "lora_train-set_cd1to5.csv"

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
)

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

VISION_EXCLUDE = r".*(vision_tower|multi_modal_projector).*"

PAD_ID = 0


def encode(tokenizer, user_prompt, assistant=None, eos_id=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if assistant is None:
        request = ChatCompletionRequest(
            messages=messages,
            continue_final_message=False,
        )
    else:
        request = ChatCompletionRequest(
            messages=messages + [{"role": "assistant", "content": assistant}],
            continue_final_message=True,
        )

    ids = [int(x) for x in tokenizer.encode_chat_completion(request).tokens]

    if assistant is not None and ids[-1] != eos_id:
        ids.append(int(eos_id))

    return ids


def make_dataset(df, tokenizer, text_mode, target_mode, max_prompt_tokens, max_seq_length, eos_id):
    text_col = "cr_text_cleaned" if text_mode == "cleaned" else "cr_text_brut"
    samples = []

    for _, row in df.iterrows():
        text = str(row[text_col]).strip()

        if target_mode == "twostep":
            labels = json.loads(str(row["json_manuel"]))
            completion = json.dumps(
                {
                    key: "oui"
                    if str(labels[key]).strip().lower() in {"oui", "yes", "true", "1"}
                    else "non"
                    for key in KEYS
                },
                ensure_ascii=False,
            )
        else:
            completion = str(int(row["cd_manuel"]))

        template = TWO_STEP_PROMPT_TEMPLATE if target_mode == "twostep" else DIRECT_PROMPT_TEMPLATE
        row_data = {
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
            prompt = template.format(cr_text=candidate, **row_data)

            if len(encode(tokenizer, prompt)) <= max_prompt_tokens:
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
            chunks = [text]

        for chunk in chunks:
            prompt = template.format(cr_text=chunk, **row_data)
            prompt_ids = encode(tokenizer, prompt)
            full_ids = encode(tokenizer, prompt, completion, eos_id)
            prefix = 0

            while (
                prefix < len(prompt_ids)
                and prefix < len(full_ids)
                and prompt_ids[prefix] == full_ids[prefix]
            ):
                prefix += 1

            if len(full_ids) <= max_seq_length and 0 < prefix < len(full_ids):
                samples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": [-100] * prefix + full_ids[prefix:],
                    }
                )

    return Dataset.from_list(samples)


def collate(features):
    max_len = max(len(x["input_ids"]) for x in features)
    max_len = ((max_len + 7) // 8) * 8

    return {
        "input_ids": torch.tensor(
            [x["input_ids"] + [PAD_ID] * (max_len - len(x["input_ids"])) for x in features],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in features],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [x["labels"] + [-100] * (max_len - len(x["labels"])) for x in features],
            dtype=torch.long,
        ),
    }


def load_model(quant_mode, token):
    source = os.environ.get("MODEL_PATH", "").strip() or MODEL_ID
    common = {
        "token": token,
        "device_map": "auto",
        "attn_implementation": "eager",
    }

    if quant_mode == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Mistral3ForConditionalGeneration.from_pretrained(
            source,
            quantization_config=quantization,
            **common,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            source,
            dtype=torch.bfloat16,
            **common,
        )
        model.enable_input_require_grads()

    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    return get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
            exclude_modules=VISION_EXCLUDE,
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
    tokenizer = MistralTokenizer.from_hf_hub(source)
    inner = tokenizer.instruct_tokenizer.tokenizer
    eos_id = int(inner.eos_id)
    PAD_ID = int(inner.pad_id) if isinstance(inner.pad_id, int) and inner.pad_id >= 0 else eos_id

    if args.run_mode == "development":
        train_path = Path(args.train_csv or INTERNAL_TRAIN)
        val_path = Path(args.internal_val_csv or INTERNAL_VAL)
        epochs = float(args.num_train_epochs)
    else:
        train_path = Path(args.train_csv or FULL_TRAIN)
        val_path = None
        meta = json.loads(Path(args.selection_meta_json).read_text(encoding="utf-8"))
        epochs = float(meta["best_epoch"])

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False) if val_path else None

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

    assert required.issubset(train_df.columns)

    if val_df is not None:
        assert required.issubset(val_df.columns)
        assert not set(train_df["patient_id"].astype(str)) & set(val_df["patient_id"].astype(str))

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

    model = load_model(args.quant_mode, token)

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
        experiment_id=f"ministral3b_{args.quant_mode}_{args.text_mode}_{args.target_mode}_{args.run_mode}",
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

    trainer.save_model(str(adapter_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    best_epoch = None
    best_eval_loss = None

    if args.run_mode == "development":
        records = [
            x
            for x in trainer.state.log_history
            if x.get("eval_loss") is not None
        ]
        best = min(records, key=lambda x: float(x["eval_loss"]))
        best_epoch = float(best["epoch"])
        best_eval_loss = float(best["eval_loss"])

    metadata = {
        "model_id": MODEL_ID,
        "run_mode": args.run_mode,
        "training_role": "development_model_selection"
        if args.run_mode == "development"
        else "final_refit",
        "quant_mode": args.quant_mode,
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
        "training_wall_seconds": float(time.perf_counter() - start),
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
parser.add_argument("--quant-mode", choices=["nf4", "bf16"], default="nf4")
parser.add_argument("--max-prompt-tokens", type=int, default=5800)
parser.add_argument("--max-seq-length", type=int, default=6144)
parser.add_argument("--num-train-epochs", type=float, default=3.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

run(args)
