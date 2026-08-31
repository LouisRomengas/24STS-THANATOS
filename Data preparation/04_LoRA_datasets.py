from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

FINAL_DIR = Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/"
    "24STS-THANATOS_datasets/FINAL"
)

TRAIN_PATH = FINAL_DIR / "train_cd1to5.csv"
VAL_PATH = FINAL_DIR / "val_cd1to5.csv"

KEYS = [
    "complication_nouvelle",
    "evenement_temporel_pertinent",
    "traitement_pharmacologique_specifique",
    "geste_invasif_therapeutique",
    "usc_rea_surveillance_programmee",
    "usc_rea_non_programmee_pour_complication",
    "defaillance_organe",
    "deces_lie_complication",
]

train = pd.read_csv(TRAIN_PATH, low_memory=False)
val = pd.read_csv(VAL_PATH, low_memory=False)

for df in [train, val]:
    df["patient_id"] = df["patient_id"].fillna("").astype(str).str.strip()
    df["cd_manuel"] = pd.to_numeric(df["cd_manuel"]).astype(int)

    df["json_manuel"] = [
        json.dumps(
            {
                key: str(json.loads(value)[key]).strip().lower()
                for key in KEYS
            },
            ensure_ascii=False,
        )
        for value in df["json_manuel"]
    ]

patient_df = (
    train.groupby("patient_id", as_index=False)
    .agg(patient_grade=("cd_manuel", "max"))
)

train_ids, validation_ids = train_test_split(
    patient_df["patient_id"],
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=patient_df["patient_grade"],
)

internal_train = train[
    train["patient_id"].isin(train_ids)
].reset_index(drop=True)

internal_validation = train[
    train["patient_id"].isin(validation_ids)
].reset_index(drop=True)

train.to_csv(
    FINAL_DIR / "lora_train-set_cd1to5.csv",
    index=False,
)

internal_train.to_csv(
    FINAL_DIR / "lora_train-set_cd1to5_internal_train.csv",
    index=False,
)

internal_validation.to_csv(
    FINAL_DIR / "lora_train-set_cd1to5_internal_validation.csv",
    index=False,
)

val.to_csv(
    FINAL_DIR / "lora_val_cd1to5.csv",
    index=False,
)
