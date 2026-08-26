import hashlib
import json
import os
import re
import unicodedata

import pandas as pd

base_dir = '/export/home/cse240018/brat_data/24STS-THANATOS'
gold_standard_dir = os.path.join(base_dir, 'GoldStandard')
calibration_dir = os.path.join(base_dir, 'Calibration')

ANNOTATEURS = [
    'ADE': 50,
    'STS': 50,
    'CRE': 100,
    'ACE': 50,
[
CATEGORIES_PRINCIPALES = [
    'oesophage_estomac', 'rectum', 'colon', 'foie_pancreas_voies_biliaires'
]

PATIENTS_PAR_ANNOTATEUR = 50
PATIENTS_CALIBRATION = 30

SUBFOLDERS = [
    'CR_imagerie', 'CR_hospitalisation', 'CR_operatoire', 'CR_autres'
]

PATTERN_REPORTS = r'"reports"\s*:\s*(\[\s*{.*?}\s*\])'

ANNOTATION_CONF = """[entities]
COMPLICATION
TRAITEMENT_MEDICAL
TRAITEMENT_CHIRURGICAL
TRAITEMENT_RADIOLOGIQUE
TRAITEMENT_ENDOSCOPIQUE
TRAITEMENT_REANIMATION
DECES

[attributes]
CD_Grade Arg:TRAITEMENT_MEDICAL, Value:I|II
CD_Sous_Grade Arg:TRAITEMENT_CHIRURGICAL|TRAITEMENT_RADIOLOGIQUE|TRAITEMENT_ENDOSCOPIQUE, Value:a|b
CD_Type_Defaillance Arg:TRAITEMENT_REANIMATION, Value:a|b

[relations]
NecessiteTrait Arg1:COMPLICATION, Arg2:TRAITEMENT_MEDICAL
NecessiteChir Arg1:COMPLICATION, Arg2:TRAITEMENT_CHIRURGICAL
NecessiteEndo Arg1:COMPLICATION, Arg2:TRAITEMENT_ENDOSCOPIQUE
NecessiteRadio Arg1:COMPLICATION, Arg2:TRAITEMENT_RADIOLOGIQUE
NecessiteRea Arg1:COMPLICATION, Arg2:TRAITEMENT_REANIMATION

[events]
"""


def is_pdf_scan(text):
    if not isinstance(text, str):
        return False

    s = text.strip().strip('"')

    if s == 'FICHIER PDF SCAN':
        return True

    if 'FICHIER PDF SCAN' in s:
        if len(s) < 100:
            return True

        if (s.count('FICHIER PDF SCAN') * len('FICHIER PDF SCAN') / len(s)) > 0.5:
            return True

    return False


def parse_reports(raw):
    if not isinstance(raw, str) or is_pdf_scan(raw):
        return []

    cleaned = raw.strip()

    if cleaned.startswith('{') and cleaned.endswith('}'):
        cleaned = (
            cleaned.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        )
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', cleaned)

    try:
        data = json.loads(cleaned, strict=False)

        if isinstance(data, dict) and 'reports' in data:
            return data['reports']
    except Exception:
        m = re.search(PATTERN_REPORTS, raw, re.DOTALL)

        if m:
            try:
                return json.loads(m.group(1), strict=False)
            except Exception:
                return []

    return []


def process_patient(row, target_dir, annotateur=None):
    pid = row['person_id']
    surgery_date = row.get('major_surgery_date', '')
    surgery_type = row.get('surgery_category', '')

    if surgery_type in ['foie_pancreas', 'voies_biliaires']:
        surgery_type = 'foie_pancreas_voies_biliaires'

    safe_id = 'P' + hashlib.md5(str(pid).encode()).hexdigest()[:8]
    patient_dir = os.path.join(target_dir, safe_id)
    os.makedirs(patient_dir, exist_ok=True)

    for sf in SUBFOLDERS:
        os.makedirs(os.path.join(patient_dir, sf), exist_ok=True)

    reports = parse_reports(row.get('cr_info', ''))

    if not reports:
        return

    counters = {sf: 1 for sf in SUBFOLDERS}

    for rpt in reports:
        date = rpt.get('date', '')
        title = rpt.get('title', '')
        typ = rpt.get('type', '')

        text = rpt.get('text', '')

        if not isinstance(text, str):
            text = ''
        else:
            text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
            text = text.replace('\r\n', '\n').replace('\r', '\n')

        annule = re.sub(r'\s+', '', text.upper())

        if is_pdf_scan(text) or any(
            p in annule
            for p in [
                'DOCUMENTANNULE', 'DOCUMENTANNULÉ', 'RAPPORTANNULE', 'CRANNULÉ'
            ]
        ):
            continue

        lower_title = (title or '').lower()
        upper_type = (typ or '').upper()

        if upper_type in ['CRH-HOSPI', 'CRH-J', 'CRH-S', 'CRH-CHIR', 'SYNTH'] or any(
            k in lower_title for k in ['hospit', 'sortie', 'séjour', 'sejour']
        ):
            sub = 'CR_hospitalisation'
        elif upper_type in ['CR-OPER', 'CR-ACTE-DIAG-AUTRE'] or any(
            k in lower_title for k in ['operatoire', 'chir', 'bloc']
        ):
            sub = 'CR_operatoire'
        elif upper_type == 'CR-IMAGE' or any(
            k in lower_title for k in ['image', 'scanner', 'irm', 'echo', 'radio']
        ):
            sub = 'CR_imagerie'
        else:
            sub = 'CR_autres'

        idx = counters[sub]
        counters[sub] += 1

        if not title or title.lower() == 'unknown_title':
            base = f"NO_NAME_{idx:02d}"
        else:
            base = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            base = ''.join(
                c for c in unicodedata.normalize('NFD', base)
                if unicodedata.category(c) != 'Mn'
            )
            base = base[:30] + f"_{idx:02d}"

        if annotateur is None:
            ligne_usage = "USAGE: CALIBRATION\n"
        else:
            ligne_usage = f"ANNOTATEUR ASSIGNÉ: {annotateur}\n"

        header = (
            f"ID PATIENT ORIGINAL: {pid}\n"
            f"DATE DE CHIRURGIE: {surgery_date}\n"
            f"TYPE DE CHIRURGIE: {surgery_type}\n"
            + ligne_usage
            + f"DATE DU RAPPORT: {date}\n"
            f"TITRE DU RAPPORT: {title}\n"
            f"TYPE DU RAPPORT: {typ}\n"
            f"CLASSÉ DANS: {sub}\n"
            + ("=" * 50) + "\n\n"
        )

        with open(
            os.path.join(patient_dir, sub, f"{base}.txt"), 'w', encoding='utf-8'
        ) as f:
            f.write(header + text)

        open(
            os.path.join(patient_dir, sub, f"{base}.ann"), 'w', encoding='utf-8'
        ).close()


def select_balanced(df, n_patients):
    n_patients = min(n_patients, len(df))

    per_category = n_patients // len(CATEGORIES_PRINCIPALES)
    remaining = n_patients % len(CATEGORIES_PRINCIPALES)

    selected = []

    for i, cat in enumerate(CATEGORIES_PRINCIPALES):
        cat_patients = df[df['surgery_category'] == cat]
        target = per_category + (1 if i < remaining else 0)

        selected.append(
            cat_patients.sample(n=min(target, len(cat_patients)), random_state=42)
        )

    balanced = pd.concat(selected)

    if len(balanced) < n_patients:
        missing = n_patients - len(balanced)
        pool = df[~df.index.isin(balanced.index)]

        if len(pool) >= missing:
            balanced = pd.concat([
                balanced, pool.sample(n=missing, random_state=42)
            ])

    elif len(balanced) > n_patients:
        balanced = balanced.sample(n=n_patients, random_state=42)

    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def read_original_ids(root):
    used_patients = set()

    if not os.path.exists(root):
        return used_patients

    for annotateur in os.listdir(root):
        annotateur_path = os.path.join(root, annotateur)

        if not os.path.isdir(annotateur_path):
            continue

        for patient_dir in os.listdir(annotateur_path):
            patient_path = os.path.join(annotateur_path, patient_dir)

            if not os.path.isdir(patient_path) or not patient_dir.startswith('P'):
                continue

            for subfolder in SUBFOLDERS:
                subfolder_path = os.path.join(patient_path, subfolder)

                if not os.path.exists(subfolder_path):
                    continue

                for file in os.listdir(subfolder_path):
                    if not file.endswith('.txt'):
                        continue

                    try:
                        with open(
                            os.path.join(subfolder_path, file),
                            'r', encoding='utf-8'
                        ) as f:
                            first_line = f.readline().strip()

                        if first_line.startswith('ID PATIENT ORIGINAL: '):
                            used_patients.add(
                                first_line.replace('ID PATIENT ORIGINAL: ', '')
                            )
                            break
                    except Exception:
                        continue

                break

    return used_patients


if __name__ == '__main__':
    df = final_cases.toPandas()

    df['surgery_category'] = df['surgery_category'].apply(
        lambda x: 'foie_pancreas_voies_biliaires'
        if x in ['foie_pancreas', 'voies_biliaires'] else x
    )

    df['n_reports'] = df['cr_info'].apply(lambda raw: len(parse_reports(raw)))
    df = df[df['n_reports'] > 0]
    df = df[df['surgery_category'].isin(CATEGORIES_PRINCIPALES)]

    os.makedirs(gold_standard_dir, exist_ok=True)

    with open(
        os.path.join(gold_standard_dir, 'annotation.conf'), 'w', encoding='utf-8'
    ) as f:
        f.write(ANNOTATION_CONF)

    gold_patients = select_balanced(
        df, sum(ANNOTATEURS.values())
    )

    start = 0

    for annotateur, n_patients in ANNOTATEURS.items():
        annotateur_dir = os.path.join(gold_standard_dir, annotateur)
        os.makedirs(annotateur_dir, exist_ok=True)

        with open(
            os.path.join(annotateur_dir, 'annotation.conf'), 'w', encoding='utf-8'
        ) as f:
            f.write(ANNOTATION_CONF)

        assigned = gold_patients.iloc[start:start + n_patients]
        start += n_patients

        for _, row in assigned.iterrows():
            process_patient(row, annotateur_dir, annotateur)

    os.makedirs(calibration_dir, exist_ok=True)

    with open(
        os.path.join(calibration_dir, 'annotation.conf'), 'w', encoding='utf-8'
    ) as f:
        f.write(ANNOTATION_CONF)

    used_patients = read_original_ids(gold_standard_dir)

    calibration_patients = select_balanced(
        df[~df['person_id'].isin(used_patients)], PATIENTS_CALIBRATION
    )

    for _, row in calibration_patients.iterrows():
        process_patient(row, calibration_dir)
