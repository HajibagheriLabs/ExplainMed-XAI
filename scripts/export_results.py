"""Assemble the final model comparison, reports/results.csv, from the committed reports."""

import argparse

import pandas as pd

from explainmed.config import load_config

MODELS = ["image_only", "text_only", "fusion", "fusion_minus_image_only"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    reports_dir = cfg.paths.reports_dir

    classification = pd.read_csv(reports_dir / "fusion_results.csv", index_col=0)
    missing = set(MODELS) - set(classification.index)
    if missing:
        raise ValueError(
            f"fusion_results.csv lacks {sorted(missing)}; run `make train`"
        )
    # values are copied as written, so this table and its sources agree to the digit
    results = classification.loc[MODELS]
    results.to_csv(
        reports_dir / "results.csv", float_format="%.4f", lineterminator="\n"
    )
    print(results.T.to_string())


if __name__ == "__main__":
    main()
