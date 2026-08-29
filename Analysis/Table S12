import json
from pathlib import Path

import pandas as pd

BASE = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_results"
)
OUT = BASE / "Performances"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING = BASE / "embeddings_qwen3e8b_cd1to5"

ZERO_SHOT = [
    ("Gemma 3 4B", "google/gemma-3-4b-it", BASE / "llm_gemma3_4b_bf16_cd1to5"),
    ("MedGemma 4B", "google/medgemma-4b-it", BASE / "llm_medgemma4b_bf16_cd1to5"),
    ("Ministral 3B", "mistralai/Ministral-3-3B-Instruct-2512-BF16", BASE / "llm_ministral3b_bf16_cd1to5"),
    ("Gemma 3 27B", "google/gemma-3-27b-it", BASE / "llm_gemma3_27b_bf16_cd1to5"),
    ("MedGemma 27B", "google/medgemma-27b-text-it", BASE / "llm_medgemma27b_bf16_cd1to5"),
    ("Mistral Small 3.2 24B", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", BASE / "llm_mistral24b_bf16_cd1to5"),
]

QLORA = [
    ("Gemma 3 4B", "google/gemma-3-4b-it", BASE / "gemma3_4b_qlora_traineval_cd1to5"),
    ("MedGemma 4B", "google/medgemma-4b-it", BASE / "medgemma_4b_qlora_traineval_cd1to5"),
    ("Ministral 3B", "mistralai/Ministral-3-3B-Instruct-2512-BF16", BASE / "lora_ministral3b_nf4_cd1to5"),
    ("Gemma 3 27B", "google/gemma-3-27b-it", BASE / "gemma3_27b_qlora_traineval_cd1to5"),
    ("MedGemma 27B", "google/medgemma-27b-text-it", BASE / "medgemma27b_qlora_traineval_cd1to5"),
    ("Mistral Small 3.2 24B", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", BASE / "mistral24b_qlora_traineval_cd1to5"),
]

TEXT = {"raw": "Raw", "cleaned": "Cleaned"}
STRATEGY = {"direct": "One-step", "twostep": "Two-step"}
STAGE = {
    "encoding": "Computation",
    "training": "Classifier selection and fitting",
}

rows = []

assert EMBEDDING.is_dir()
embedding_summaries = sorted(EMBEDDING.rglob("paper_summary.json"))
assert len(embedding_summaries) == 2

for path in embedding_summaries:
    summary = json.loads(path.read_text(encoding="utf-8"))
    text_mode = "cleaned" if "cleaned" in str(summary["text_col"]) else "raw"
    energy_report = summary["energy_report"]

    for key in ("encoding", "training"):
        values = energy_report[key]["codecarbon"]
        rows.append(
            {
                "_order": 0,
                "_stage": 0 if key == "encoding" else 1,
                "Model": "Qwen3-Embedding-8B",
                "Method": "Logistic regression",
                "Stage": STAGE[key],
                "Text": TEXT[text_mode],
                "Strategy": "N/A",
                "Duration (h)": float(values["duration_seconds"]) / 3600,
                "Energy (kWh)": float(values["energy_kwh"]),
                "Emissions (kg CO2eq)": float(values["emissions_kg_co2eq"]),
            }
        )

for model_order, (model, model_id, directory) in enumerate(ZERO_SHOT, start=1):
    assert directory.is_dir()
    summaries = sorted(directory.rglob("paper_summary.json"))
    assert len(summaries) == 4

    seen = set()

    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["model_id"] == model_id
        text_mode = (
            summary.get("text_mode")
            or summary.get("text_tag")
            or ("cleaned" if "cleaned" in str(summary["text_col"]) else "raw")
        )
        target_mode = summary.get("target_mode") or summary.get("mode")
        assert text_mode in TEXT
        assert target_mode in STRATEGY
        assert (text_mode, target_mode) not in seen
        seen.add((text_mode, target_mode))
        values = summary["energy"]

        rows.append(
            {
                "_order": model_order,
                "_stage": 2,
                "Model": model,
                "Method": "Zero-shot",
                "Stage": "Inference",
                "Text": TEXT[text_mode],
                "Strategy": STRATEGY[target_mode],
                "Duration (h)": float(values["duration_seconds"]) / 3600,
                "Energy (kWh)": float(values["energy_kwh"]),
                "Emissions (kg CO2eq)": float(values["emissions_kg_co2eq"]),
            }
        )

    assert seen == {
        ("cleaned", "direct"),
        ("cleaned", "twostep"),
        ("raw", "direct"),
        ("raw", "twostep"),
    }

for model_order, (model, model_id, directory) in enumerate(QLORA, start=7):
    assert directory.is_dir()

    training = {}

    for meta_path in sorted(directory.rglob("training_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("model_id") != model_id:
            continue

        role = meta.get("training_role")
        text_mode = meta.get("text_mode")
        target_mode = meta.get("target_mode")

        if role not in {"development_model_selection", "final_refit"}:
            continue

        assert text_mode in TEXT
        assert target_mode in STRATEGY
        key = (text_mode, target_mode, role)
        assert key not in training

        emissions_path = meta_path.parent / "emissions.csv"
        assert emissions_path.is_file()
        emissions = pd.read_csv(emissions_path, low_memory=False)
        assert not emissions.empty

        if "timestamp" in emissions.columns:
            timestamps = pd.to_datetime(emissions["timestamp"], errors="coerce")
            row = emissions.loc[timestamps.idxmax()] if timestamps.notna().any() else emissions.iloc[-1]
        else:
            row = emissions.iloc[-1]

        training[key] = {
            "duration_seconds": float(row["duration"]),
            "energy_kwh": float(row["energy_consumed"]),
            "emissions_kg_co2eq": float(row["emissions"]),
        }

    summaries = []

    for summary_path in sorted(directory.rglob("paper_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        if summary.get("method") != "qlora" or summary.get("model_id") != model_id:
            continue

        summaries.append((summary_path, summary))

    assert len(summaries) == 4
    seen = set()

    for summary_path, summary in summaries:
        text_mode = summary["text_mode"]
        target_mode = summary["target_mode"]
        assert text_mode in TEXT
        assert target_mode in STRATEGY
        assert (text_mode, target_mode) not in seen
        seen.add((text_mode, target_mode))

        development = training[(text_mode, target_mode, "development_model_selection")]
        refit = training[(text_mode, target_mode, "final_refit")]

        rows.append(
            {
                "_order": model_order,
                "_stage": 1,
                "Model": model,
                "Method": "QLoRA",
                "Stage": "Training",
                "Text": TEXT[text_mode],
                "Strategy": STRATEGY[target_mode],
                "Duration (h)": (
                    development["duration_seconds"] + refit["duration_seconds"]
                ) / 3600,
                "Energy (kWh)": development["energy_kwh"] + refit["energy_kwh"],
                "Emissions (kg CO2eq)": (
                    development["emissions_kg_co2eq"]
                    + refit["emissions_kg_co2eq"]
                ),
            }
        )

        emissions_path = summary_path.parent / "emissions.csv"
        assert emissions_path.is_file()
        emissions = pd.read_csv(emissions_path, low_memory=False)
        assert not emissions.empty

        if "timestamp" in emissions.columns:
            timestamps = pd.to_datetime(emissions["timestamp"], errors="coerce")
            row = emissions.loc[timestamps.idxmax()] if timestamps.notna().any() else emissions.iloc[-1]
        else:
            row = emissions.iloc[-1]

        rows.append(
            {
                "_order": model_order,
                "_stage": 2,
                "Model": model,
                "Method": "QLoRA",
                "Stage": "Inference",
                "Text": TEXT[text_mode],
                "Strategy": STRATEGY[target_mode],
                "Duration (h)": float(row["duration"]) / 3600,
                "Energy (kWh)": float(row["energy_consumed"]),
                "Emissions (kg CO2eq)": float(row["emissions"]),
            }
        )

    assert seen == {
        ("cleaned", "direct"),
        ("cleaned", "twostep"),
        ("raw", "direct"),
        ("raw", "twostep"),
    }

table = (
    pd.DataFrame(rows)
    .sort_values(
        ["_order", "_stage", "Text", "Strategy"],
        kind="stable",
    )
    .drop(columns=["_order", "_stage"])
    .reset_index(drop=True)
)

table["Duration (h)"] = table["Duration (h)"].round(3)
table["Energy (kWh)"] = table["Energy (kWh)"].round(6)
table["Emissions (kg CO2eq)"] = table["Emissions (kg CO2eq)"].round(6)

table.to_csv(
    OUT / "CodeCarbon.csv",
    index=False,
    encoding="utf-8-sig",
)
