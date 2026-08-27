import pathlib
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd

GS_PATH = pathlib.Path(
    "/export/home/cse240018/brat_data/24STS-THANATOS/GoldStandard"
)

CORRESP_PATH = pathlib.Path(
    "/export/home/cse240018/24STS-THANATOS_LRS/24STS-THANATOS_datasets"
    "/correspondance_ids_20250605_121018.csv"
)

EXCLUDED_PATIENTS = {
    "P4a1ff14e", "P93c237b6", "Pd1b10550",
    "P3295391d", "P40143be9", "P26977228",
    "P11affc0f", "P72be35db", "Pa03bf529",
}

HEADER_FIELDS = {
    'id_patient_original': r'ID PATIENT ORIGINAL:\s*(.+)',
    'date_chirurgie':
        r'DATE DE CHIRURGIE:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})',
    'type_chirurgie': r'TYPE DE CHIRURGIE:\s*(.+)',
    'annotateur': r'ANNOTATEUR ASSIGNÉ:\s*(.+)',
    'date_rapport':
        r'DATE DU RAPPORT:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})',
    'titre_rapport': r'TITRE DU RAPPORT:\s*(.+)',
    'type_rapport': r'TYPE DU RAPPORT:\s*(.+)',
}

GRADE_ORDER = ['1', '2', '3a', '3b', '4a', '4b', '5']

TRAITEMENTS_CD3 = {
    "TRAITEMENT_CHIRURGICAL",
    "TRAITEMENT_RADIOLOGIQUE",
    "TRAITEMENT_ENDOSCOPIQUE",
}

DOC_ANNULE_RX = re.compile(r"D\W*O\W*C\W*U\W*M\W*E\W*N\W*T")
ANNULE_RX = re.compile(r"A\W*N\W*N\W*U\W*L\W*[ÉE]")


def parse_date(value):
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue

    return None


def extract_cr_info_from_header(cr_text):
    header_lines = []

    for line in cr_text.split('\n'):
        if line.strip().startswith("===="):
            break

        header_lines.append(line)

    header_text = '\n'.join(header_lines)
    header_info = {}

    for key, pattern in HEADER_FIELDS.items():
        m = re.search(pattern, header_text)

        if m:
            header_info[key] = m.group(1).strip()

    for key in ("date_chirurgie", "date_rapport"):
        if key in header_info:
            header_info[key] = parse_date(header_info[key])

    return header_info


def parse_brat_file(ann_path):
    entities = []
    attributes = []

    try:
        lines = ann_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return entities, attributes

    for ln in lines:
        ln = ln.strip()

        if not ln:
            continue

        parts = ln.split("\t")

        if len(parts) < 2:
            continue

        if ln.startswith("T"):
            tinfo = parts[1].split()
            nums = [int(x) for x in re.findall(r"\d+", " ".join(tinfo[1:]))]

            if not nums:
                continue

            entities.append({
                "id": parts[0],
                "type": tinfo[0],
                "start": min(nums),
                "end": max(nums),
                "text": parts[2] if len(parts) > 2 else "",
            })

        elif ln.startswith("A"):
            ainfo = parts[1].split()

            if len(ainfo) < 3:
                continue

            attributes.append({
                "id": parts[0],
                "type": ainfo[0],
                "entity_id": ainfo[1],
                "value": ainfo[2],
            })

    return entities, attributes


def cd_grade_for_cr(entities, attributes):
    if not entities:
        return "1"

    attr_by_ent = defaultdict(dict)

    for a in attributes:
        if a["type"] in ("CD_Grade", "CD_Sous_Grade"):
            attr_by_ent[a["entity_id"]][a["type"]] = a["value"]

    for e in entities:
        if e["type"] == "DECES":
            return "5"

    cr_grade = "1"

    for e in entities:
        et = e["type"]

        if not et.startswith("TRAITEMENT_"):
            continue

        attrs = attr_by_ent.get(e["id"], {})

        if et == "TRAITEMENT_REANIMATION":
            grade = "4b" if attrs.get("CD_Sous_Grade", "a") == "b" else "4a"
        elif et in TRAITEMENTS_CD3:
            grade = "3b" if attrs.get("CD_Sous_Grade", "a") == "b" else "3a"
        elif et == "TRAITEMENT_MEDICAL":
            grade = "2" if attrs.get("CD_Grade", "I") == "II" else "1"
        else:
            continue

        if GRADE_ORDER.index(grade) > GRADE_ORDER.index(cr_grade):
            cr_grade = grade

    return cr_grade


def is_document_annule(text):
    if not isinstance(text, str):
        return False

    t = text.upper()

    return bool(DOC_ANNULE_RX.search(t)) and bool(ANNULE_RX.search(t))


def iter_patient_dirs(annotator_root):
    for entry in annotator_root.iterdir():
        if not entry.is_dir():
            continue

        if entry.name.startswith("nouveaux_patients_"):
            for inner in entry.iterdir():
                if inner.is_dir():
                    yield inner
        else:
            yield entry


corr = pd.read_csv(CORRESP_PATH)

corr_dict = {
    row["id_anonymise"]: datetime.strptime(row["date_chirurgie"], "%Y-%m-%d")
    for _, row in corr.iterrows()
}

rows = []

for annot_dir in sorted(GS_PATH.iterdir()):
    if not annot_dir.is_dir():
        continue

    for patient_dir in iter_patient_dirs(annot_dir):
        patient_id = patient_dir.name

        if patient_id in EXCLUDED_PATIENTS:
            continue

        cr_path = patient_dir / "CR_hospitalisation"

        if not cr_path.exists():
            continue

        for txt_file in sorted(cr_path.glob("*.txt")):
            try:
                cr_text = txt_file.read_text(encoding="utf-8")
            except Exception:
                continue

            if is_document_annule(cr_text):
                continue

            header = extract_cr_info_from_header(cr_text)
            cr_date = header.get("date_rapport")

            if cr_date is None:
                continue

            surgery_date = (
                header.get("date_chirurgie") or corr_dict.get(patient_id)
            )

            if surgery_date is None:
                continue

            ann_file = txt_file.with_suffix(".ann")
            ann_content = (
                ann_file.read_text(encoding="utf-8").strip()
                if ann_file.exists() else ""
            )

            if ann_content:
                entities, attributes = parse_brat_file(ann_file)
                cd_detail = cd_grade_for_cr(entities, attributes)
            else:
                cd_detail = "1"

            rows.append({
                "patient_id": patient_id,
                "fichier_cr": txt_file.name,
                "date_chirurgie": surgery_date,
                "date_cr": cr_date,
                "jours_post_chirurgie": (cr_date - surgery_date).days,
                "categorie_chirurgie": header.get("type_chirurgie", "unknown"),
                "annotateur": header.get("annotateur", annot_dir.name),
                "source": annot_dir.name,
                "cr_text_brut": cr_text,
                "cd_manuel": int(cd_detail[0]),
                "cd_manuel_detail": cd_detail,
                "has_manual_annotations": bool(ann_content),
                "ann_content": ann_content,
            })

reports_df = pd.DataFrame(rows)
