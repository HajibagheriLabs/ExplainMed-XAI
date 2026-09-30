"""Report class balance, images per lesion, age, sex, and localisation of HAM10000."""

import argparse

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

from explainmed.config import load_config
from explainmed.data import (
    CLASS_NAMES,
    SPLITS,
    assign_splits,
    load_metadata,
    load_splits,
)
from explainmed.style import BAR_LABEL_STYLE, FIGURE_STYLE, SERIES, SURFACE


def tally(frame: pd.DataFrame, column: str, statistic: str, split: str) -> pd.DataFrame:
    """Image and lesion counts per value of `column`, a lesion counted once."""
    lesions = frame.drop_duplicates("lesion_id")
    table = pd.DataFrame(
        {
            "images": frame[column].value_counts(),
            "lesions": lesions[column].value_counts(),
        }
    )
    table = table.fillna(0).astype(int).rename_axis("value").reset_index()
    table["image_fraction"] = table["images"] / len(frame)
    table["lesion_fraction"] = table["lesions"] / len(lesions)
    table.insert(0, "split", split)
    table.insert(0, "statistic", statistic)
    return table.sort_values("images", ascending=False)


def dataset_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.assign(
        total="all",
        age=metadata["age"].map(lambda a: "unknown" if pd.isna(a) else str(int(a))),
        images_per_lesion=metadata.groupby("lesion_id")["image_id"]
        .transform("size")
        .astype(str),
    )
    tables = []
    for split in ("all", *SPLITS):
        subset = frame if split == "all" else frame[frame["split"] == split]
        tables.append(tally(subset, "total", "total", split))
        tables.append(tally(subset, "dx", "class", split))
    for column in ("images_per_lesion", "age"):
        ordered = tally(frame, column, column, "all")
        numeric = pd.to_numeric(ordered["value"], errors="coerce")
        tables.append(ordered.iloc[numeric.argsort()])
    for column in ("sex", "localization"):
        tables.append(tally(frame, column, column, "all"))
    return pd.concat(tables, ignore_index=True)


def select(stats: pd.DataFrame, statistic: str) -> pd.DataFrame:
    rows = stats[(stats["statistic"] == statistic) & (stats["split"] == "all")]
    return rows.set_index("value")


def plot_overview(stats: pd.DataFrame, metadata: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 8.5), layout="constrained")
    (ax_class, ax_lesion), (ax_age, ax_site) = fig.subplots(2, 2)

    classes = select(stats, "class").iloc[::-1]
    positions = range(len(classes))
    for offset, column, color in (
        (0.19, "images", SERIES[0]),
        (-0.19, "lesions", SERIES[1]),
    ):
        bars = ax_class.barh(
            [p + offset for p in positions],
            classes[column],
            height=0.34,
            color=color,
            label=column,
        )
        ax_class.bar_label(bars, fmt="{:,.0f}", **BAR_LABEL_STYLE)
    ax_class.set_yticks(positions, [f"{CLASS_NAMES[c]} ({c})" for c in classes.index])
    ax_class.set_title("Images and lesions per diagnosis class")
    ax_class.legend(loc="lower right")

    sizes = select(stats, "images_per_lesion").sort_index()
    bars = ax_lesion.bar(sizes.index, sizes["lesions"], width=0.5, color=SERIES[0])
    ax_lesion.bar_label(bars, fmt="{:,.0f}", **BAR_LABEL_STYLE)
    ax_lesion.set_title("Lesions by number of images of the same lesion")
    ax_lesion.set_xlabel("images per lesion")
    ax_lesion.set_ylabel("lesions")

    lesions = metadata.drop_duplicates("lesion_id")
    known = lesions[lesions["sex"].isin(["male", "female"]) & lesions["age"].notna()]
    by_age = known.groupby(["age", "sex"]).size().unstack(fill_value=0)
    for sex, color in zip(("male", "female"), SERIES):
        ax_age.plot(
            by_age.index,
            by_age[sex],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=sex,
        )
    ax_age.set_title("Lesions by patient age and sex")
    hidden = len(lesions) - len(known)
    ax_age.set_xlabel(
        f"age in years ({hidden} lesions of unknown age or sex not shown)"
    )
    ax_age.set_ylim(bottom=0)
    ax_age.set_ylabel("lesions")
    ax_age.legend(loc="upper left")

    sites = select(stats, "localization").sort_values("lesions")
    bars = ax_site.barh(sites.index, sites["lesions"], height=0.6, color=SERIES[0])
    ax_site.bar_label(bars, fmt="{:,.0f}", **BAR_LABEL_STYLE)
    ax_site.set_title("Lesions by body localisation")

    for ax in (ax_class, ax_site):
        ax.grid(axis="x")
        ax.margins(x=0.12)
        ax.tick_params(axis="y", length=0)
    for ax in (ax_lesion, ax_age):
        ax.grid(axis="y")
        ax.margins(y=0.12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    metadata["split"] = assign_splits(metadata, load_splits(cfg))
    stats = dataset_stats(metadata)

    figures_dir = cfg.paths.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(
        cfg.paths.reports_dir / "dataset_stats.csv",
        index=False,
        float_format="%.4f",
        lineterminator="\n",
    )
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = plot_overview(stats, metadata)
        fig.savefig(figures_dir / "dataset_overview.png", dpi=150)
    print(stats[stats["split"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
