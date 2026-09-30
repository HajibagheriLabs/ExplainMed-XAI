"""Write the lesion-grouped splits and build the resized image cache."""

import argparse

from explainmed.config import load_config
from explainmed.data import (
    SPLITS,
    assign_splits,
    load_images,
    load_metadata,
    make_splits,
    save_splits,
)
from explainmed.train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    device = resolve_device(cfg)

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    splits = make_splits(
        metadata, cfg.data.val_fraction, cfg.data.test_fraction, cfg.seed
    )
    assigned = assign_splits(metadata, splits)
    save_splits(splits, cfg)
    for name in SPLITS:
        print(f"{name}: {len(splits[name])} lesions, {(assigned == name).sum()} images")

    images = load_images(metadata, cfg, device)
    print(f"image cache: {tuple(images.shape)} {images.dtype}")


if __name__ == "__main__":
    main()
