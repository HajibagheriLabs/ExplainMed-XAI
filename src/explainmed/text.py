"""Templating of structured metadata into synthetic clinical-style text."""

from __future__ import annotations

import pandas as pd
import torch
from transformers import AutoTokenizer

SITE_PHRASES = {
    "abdomen": "on the abdomen",
    "acral": "on an acral site",
    "back": "on the back",
    "chest": "on the chest",
    "ear": "on the ear",
    "face": "on the face",
    "foot": "on the foot",
    "genital": "in the genital area",
    "hand": "on the hand",
    "lower extremity": "on the lower extremity",
    "neck": "on the neck",
    "scalp": "on the scalp",
    "trunk": "on the trunk",
    "upper extremity": "on the upper extremity",
    "unknown": "at an unrecorded body site",
}
DIAGNOSIS_METHOD_PHRASES = {
    "histo": "confirmed by histopathology",
    "follow_up": "established by follow-up examination",
    "consensus": "established by expert consensus",
    "confocal": "confirmed by in-vivo confocal microscopy",
}


def describe(row: pd.Series, include_diagnosis_method: bool) -> str:
    """One templated description of an image's metadata; never mentions the diagnosis."""
    age = None if pd.isna(row["age"]) else int(row["age"])
    sex = row["sex"] if row["sex"] in ("male", "female") else None
    if row["sex"] not in ("male", "female", "unknown"):
        raise ValueError(f"unexpected sex value {row['sex']!r}")

    if age is not None and sex is not None:
        patient = f"{age}-year-old {sex}"
    elif age is not None:
        patient = f"{age}-year-old patient of unrecorded sex"
    elif sex is not None:
        patient = f"{sex.capitalize()} patient of unrecorded age"
    else:
        patient = "Patient of unrecorded age and sex"

    text = (
        f"{patient} with a skin lesion {SITE_PHRASES[row['localization']]}, "
        "imaged by dermatoscopy."
    )
    if include_diagnosis_method:
        text += f" The diagnosis was {DIAGNOSIS_METHOD_PHRASES[row['dx_type']]}."
    return text


def describe_all(metadata: pd.DataFrame, include_diagnosis_method: bool) -> list[str]:
    return [describe(row, include_diagnosis_method) for _, row in metadata.iterrows()]


def tokenize(
    texts: list[str], tokenizer_name: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """Token ids, attention masks, and special-token masks, padded to the longest text."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tokenizer(
        texts,
        padding="longest",
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        # marks [CLS], [SEP], and padding, which text attributions exclude
        "special_tokens_mask": encoded["special_tokens_mask"].to(device),
    }
