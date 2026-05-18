from pathlib import Path
import re

import numpy as np
import pandas as pd


STANDARD_12_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "positive", "pos"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "negative", "neg"}


def _clean_number(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def _label_to_int(value):
    if value is None:
        return np.nan
    text = str(value).strip().strip("\"'").lower()
    if text in _TRUE_VALUES:
        return 1
    if text in _FALSE_VALUES:
        return 0
    numeric = _clean_number(text)
    if pd.isna(numeric):
        return np.nan
    return int(numeric != 0)


def _normalize_sex(value):
    if value is None:
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"m", "male", "man"}:
        return "Male"
    if text in {"f", "female", "woman"}:
        return "Female"
    return "Unknown"


def _parse_header(record_path):
    lines = Path(str(record_path) + ".hea").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    first = lines[0].split()
    record_name = first[0]
    n_leads = int(first[1]) if len(first) > 1 else np.nan
    fs_hz = _clean_number(first[2]) if len(first) > 2 else np.nan
    n_samples_raw = _clean_number(first[3]) if len(first) > 3 else np.nan
    n_samples = int(n_samples_raw) if not pd.isna(n_samples_raw) else np.nan

    signal_lines = [ln for ln in lines[1:] if ln.strip() and not ln.startswith("#")]
    if not pd.isna(n_leads):
        lead_names = [ln.split()[-1] for ln in signal_lines[: int(n_leads)]]
    else:
        lead_names = []
    lead_names_norm = [name.strip().upper() for name in lead_names]
    has_standard_12 = set(STANDARD_12_LEADS).issubset(set(lead_names_norm))

    comments = {}
    for line in lines:
        if not line.startswith("#"):
            continue
        body = line[1:].strip()
        key, sep, value = body.partition(":")
        if sep:
            comments[key.strip().lower()] = value.strip()

    age = _clean_number(comments.get("age"))
    sex = _normalize_sex(comments.get("sex"))
    label = _label_to_int(comments.get("chagas label"))
    source = comments.get("source")

    duration_sec = (float(n_samples) / float(fs_hz)
                    if not pd.isna(n_samples) and not pd.isna(fs_hz) else np.nan)

    return {
        "record_name": record_name,
        "source": source,
        "age": age,
        "sex": sex,
        "sex_code": {"Male": 1, "Female": 0}.get(sex, np.nan),
        "fs_hz": fs_hz,
        "n_samples": n_samples,
        "duration_sec": duration_sec,
        "n_leads": n_leads,
        "lead_names": lead_names,
        "has_standard_12_leads": bool(has_standard_12),
        "chagas_label": label,
    }


def find_records(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p.with_suffix("") for p in folder.rglob("*.hea"))


def build_metadata(dataset_groups, tqdm_func=None):
    rows = []
    for group, folders in dataset_groups.items():
        for folder in folders:
            records = find_records(folder)
            iterator = records if tqdm_func is None else tqdm_func(
                records, desc=f"{group}/{Path(folder).name}"
            )
            for record_path in iterator:
                row = _parse_header(record_path)
                row.update({
                    "dataset_group": group,
                    "record_path": str(record_path),
                })
                if row["source"] is None:
                    row["source"] = group
                rows.append(row)
    return pd.DataFrame(rows)


def label_summary(df, label_col="chagas_label"):
    return {
        "total_labels": int(df[label_col].notna().sum()),
        "negative_labels": int((df[label_col] == 0).sum()),
        "positive_labels": int((df[label_col] == 1).sum()),
        "missing_labels": int(df[label_col].isna().sum()),
    }
