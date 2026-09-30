"""Write reports/text_examples.md showing exactly what the text modality contains."""

import argparse

import pandas as pd

from explainmed.config import load_config
from explainmed.data import (
    CLASS_NAMES,
    CLASSES,
    assign_splits,
    load_metadata,
    load_splits,
)
from explainmed.evaluate import classification_metrics
from explainmed.text import DIAGNOSIS_METHOD_PHRASES, SITE_PHRASES, describe


def markdown_table(frame: pd.DataFrame) -> str:
    cells = frame.astype(str).to_numpy().tolist()
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


def pick_examples(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One image per class, plus the first images with missing values."""
    per_class = [
        metadata[metadata["dx"] == dx].sample(1, random_state=seed) for dx in CLASSES
    ]
    missing = [
        metadata[metadata["age"].isna()].head(1),
        metadata[metadata["sex"] == "unknown"].head(1),
        metadata[metadata["localization"] == "unknown"].head(1),
    ]
    return pd.concat(per_class + missing).drop_duplicates("image_id")


def lookup_table_reference(
    metadata: pd.DataFrame, texts: pd.Series, split: pd.Series
) -> dict[str, float]:
    """Test metrics of memorising each training description's class frequencies."""
    train = split.to_numpy() == "train"
    test = split.to_numpy() == "test"
    labels = metadata["label"].to_numpy()
    counts = pd.crosstab(texts[train].to_numpy(), labels[train])
    counts = counts.reindex(columns=range(len(CLASSES)), fill_value=0)
    # dividing by class size makes the rule target balanced accuracy, as the models do
    per_class_rate = counts / counts.sum(axis=0)
    table = per_class_rate.idxmax(axis=1)
    most_common = int(pd.Series(labels[train]).mode()[0])
    predicted = texts[test].map(table).fillna(most_common).astype(int).to_numpy()
    return classification_metrics(labels[test], predicted, CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)

    examples = pick_examples(metadata, cfg.seed)
    example_table = pd.DataFrame(
        {
            "Image": examples["image_id"],
            "Diagnosis (the label, never in the text)": [
                f"{CLASS_NAMES[dx]} (`{dx}`)" for dx in examples["dx"]
            ],
            "Text the model reads": [
                describe(row, include_diagnosis_method=False)
                for _, row in examples.iterrows()
            ],
            "Sentence added in the diagnosis-method variant": [
                describe(row, True).removeprefix(describe(row, False)).strip()
                for _, row in examples.iterrows()
            ],
        }
    )

    descriptions = {
        include: pd.Series([describe(row, include) for _, row in metadata.iterrows()])
        for include in (False, True)
    }
    distinct = {include: texts.nunique() for include, texts in descriptions.items()}
    classes_per_text = metadata["dx"].groupby(descriptions[False].to_numpy()).nunique()
    shared = descriptions[False].map(classes_per_text).gt(1).mean()
    split = assign_splits(metadata, load_splits(cfg))
    reference = {
        include: lookup_table_reference(metadata, texts, split)
        for include, texts in descriptions.items()
    }
    crosstab = pd.crosstab(metadata["dx_type"], metadata["dx"]).reindex(
        columns=list(CLASSES), fill_value=0
    )
    single_class = [
        f"- every image whose diagnosis was {DIAGNOSIS_METHOD_PHRASES[method]} is "
        f"`{counts.idxmax()}` ({counts.max():,} images)"
        for method, counts in crosstab.iterrows()
        if (counts > 0).sum() == 1
    ]
    histo_only = [dx for dx in CLASSES if crosstab[dx].drop("histo").sum() == 0]

    template_rows = pd.DataFrame(
        {"`localization` value": list(SITE_PHRASES), "Phrase": SITE_PHRASES.values()}
    )
    method_rows = pd.DataFrame(
        {
            "`dx_type` value": list(DIAGNOSIS_METHOD_PHRASES),
            "Sentence": [
                f"The diagnosis was {phrase}."
                for phrase in DIAGNOSIS_METHOD_PHRASES.values()
            ],
        }
    )

    report = f"""# The text modality

HAM10000 has no clinical notes. The text branch of every model in this repo reads a sentence
built by a fixed template from the structured metadata of each image. It is synthetic, it is not
clinical documentation, and results on it say nothing about real clinical text.

This file is written by `scripts/text_examples.py`; the template itself is
`src/explainmed/text.py`.

## The template

```
<patient> with a skin lesion <site>, imaged by dermatoscopy.[ The diagnosis was <method>.]
```

`<patient>` is `<age>-year-old <sex>` when both are recorded, and says "unrecorded age" or
"unrecorded sex" otherwise. The diagnosis itself is never part of the text.

{markdown_table(template_rows)}

The bracketed diagnosis-method sentence is only present in one ablation (see below):

{markdown_table(method_rows)}

## Examples

One image per class, sampled with seed {cfg.seed}, plus the first images with a missing age, sex, or
body site.

{markdown_table(example_table)}

## Why the diagnosis method is left out

The diagnosis method records how the ground truth was established, which depends on the diagnosis
itself: suspicious lesions are excised and go to histopathology, benign-looking ones are followed
up. It would not be known when a prediction is made, and it nearly determines the label:

{chr(10).join(single_class)}
- every image of {", ".join(f"`{dx}`" for dx in histo_only)} was confirmed by histopathology

Images per diagnosis method and diagnosis:

{markdown_table(crosstab.rename_axis(None).reset_index(names="`dx_type`"))}

The main text modality therefore excludes it. The text-only baseline is trained both with and
without it, so the size of the leak is measured instead of hidden.

## How much the text can carry

The {len(metadata):,} images map to only {distinct[False]:,} distinct descriptions without the
diagnosis method ({distinct[True]:,} with it). {shared:.0%} of images have a description that also
belongs to an image of a different diagnosis, which caps what any text-only model can achieve.

A lookup table that memorises how often each exact training description occurs in each class, and
predicts the class where it is relatively most frequent, reaches a test macro-F1 of
{reference[False]["macro_f1"]:.4f} and a balanced accuracy of {reference[False]["balanced_accuracy"]:.4f}
({reference[True]["macro_f1"]:.4f} and {reference[True]["balanced_accuracy"]:.4f} with the
diagnosis method). This is a reference for how much a text-only model can extract, not a
baseline to beat: descriptions unseen in training fall back to the most common class.
"""
    out_file = cfg.paths.reports_dir / "text_examples.md"
    out_file.write_text(report, encoding="utf-8", newline="\n")
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
