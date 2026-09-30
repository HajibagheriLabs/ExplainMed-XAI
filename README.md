# ExplainMed-XAI

Multimodal skin-lesion classification on HAM10000 where the explanations are **measured, not
illustrated**. Grad-CAM and integrated gradients on the image branch, and attention on the text
branch, are scored with faithfulness metrics (deletion and insertion AUC against a
random-attribution control, sparsity, localisation, inter-method agreement) and put through the
Adebayo et al. model- and label-randomisation sanity checks. A method that fails a check is
reported as failing.

> [!WARNING]
> **Not a diagnostic tool.** This is a research artefact for studying whether saliency
> explanations can be trusted. It has not been clinically validated, must not be used to make or
> support decisions about any patient, and nothing it outputs is a diagnosis.

> [!IMPORTANT]
> **The text modality is synthetic.** HAM10000 contains no clinical notes. The "text" input is a
> sentence built by a fixed template from the dataset's structured metadata (age, sex, lesion
> localisation), for example *"45-year-old male with a skin lesion on the back, imaged by
> dermatoscopy."* It is not real clinical documentation, and results on it say nothing about how a
> model would behave on genuine clinical notes. See [`reports/text_examples.md`](reports/text_examples.md).

## Status

The data layer and the unimodal baselines are done; the fusion model is not trained yet. Test-split
results, mean ± sd over three seeds:

| Model      |        Macro-F1 | Balanced accuracy | Melanoma recall |
| ---------- | --------------: | ----------------: | --------------: |
| Image only | 0.7181 ± 0.0141 |   0.7199 ± 0.0143 | 0.7341 ± 0.0150 |
| Text only  | 0.2262 ± 0.0162 |   0.3164 ± 0.0296 | 0.2083 ± 0.0449 |
| Fusion     |             TBD |               TBD |             TBD |

Every number in this README is produced by a script in this repo; a result whose run does not exist
yet is written as `TBD`.

## Dataset

[HAM10000](https://doi.org/10.7910/DVN/DBW86T): 10,015 dermatoscopic images of 7,470 distinct
lesions across seven diagnostic classes, with a metadata CSV giving age, sex, lesion localisation,
and how the diagnosis was established.

| Class                                                   | Images | Lesions | Share of images |
| ------------------------------------------------------- | -----: | ------: | --------------: |
| melanocytic nevus (`nv`)                                |  6,705 |   5,403 |           66.9% |
| melanoma (`mel`)                                        |  1,113 |     614 |           11.1% |
| benign keratosis-like lesion (`bkl`)                    |  1,099 |     727 |           11.0% |
| basal cell carcinoma (`bcc`)                            |    514 |     327 |            5.1% |
| actinic keratosis / intraepithelial carcinoma (`akiec`) |    327 |     228 |            3.3% |
| vascular lesion (`vasc`)                                |    142 |      98 |            1.4% |
| dermatofibroma (`df`)                                   |    115 |      73 |            1.1% |

Two properties of this dataset shape everything downstream:

- **It is severely imbalanced.** Two thirds of the images are melanocytic nevi, so a model that
  always answers "nevus" is 67% accurate and useless. Results are reported as macro-F1, balanced
  accuracy, and per-class recall, never as accuracy alone.
- **`lesion_id` is not `image_id`.** 1,956 lesions (26%) were photographed more than once, and
  their 4,501 images make up 45% of the dataset. Images of the same lesion are near-duplicates.

![Class balance, images per lesion, age and sex, and localisation in HAM10000](reports/figures/dataset_overview.png)

The full tables behind the figure, including the class balance of each split, are in
[`reports/dataset_stats.csv`](reports/dataset_stats.csv), written by `scripts/inspect_data.py`.

Download the dataset and place it as:

```
data/raw/ham10000/HAM10000_metadata.csv
data/raw/ham10000/images/ISIC_*.jpg
```

MIMIC-CXR was deliberately not used: it requires credentialed access, so nobody could rerun this
repo end to end.

## Splits are grouped by lesion

Train, validation, and test are split **by lesion, never by image**, stratified by diagnosis, with
a fixed seed. The lesion ids of each split are committed in
[`configs/splits.json`](configs/splits.json), and a test asserts that no lesion appears in two
splits.

| Split      | Lesions | Images |
| ---------- | ------: | -----: |
| Train      |   5,228 |  7,054 |
| Validation |   1,121 |  1,464 |
| Test       |   1,121 |  1,497 |

### What happens if you split by image instead

`scripts/leakage_demo.py` measures what the grouping is worth. It fine-tunes the same ImageNet
ResNet-18 with the same hyperparameters under two protocols: the lesion-grouped split, and a naive
split that stratifies by diagnosis but ignores `lesion_id`. Both splits are redrawn and the model
retrained under five seeds.

| Mean ± sd over 5 seeds                         |  Lesion-grouped | Naive image-level |  Naive − grouped |
| ---------------------------------------------- | --------------: | ----------------: | ---------------: |
| Accuracy                                       | 0.8493 ± 0.0084 |   0.8754 ± 0.0095 | +0.0261 ± 0.0101 |
| Balanced accuracy                              | 0.6989 ± 0.0254 |   0.7644 ± 0.0238 | +0.0655 ± 0.0178 |
| Macro-F1                                       | 0.7239 ± 0.0158 |   0.7860 ± 0.0208 | +0.0621 ± 0.0231 |
| Accuracy on single-image lesions (cannot leak) | 0.9207 ± 0.0085 |   0.9238 ± 0.0092 | +0.0032 ± 0.0032 |
| Accuracy on multi-image lesions                | 0.7627 ± 0.0222 |   0.8166 ± 0.0180 | +0.0539 ± 0.0235 |

The naive split scores higher on accuracy, balanced accuracy, and macro-F1 in all five seeds.
Accuracy is inflated by about 2.6 points, and the metrics that matter on an imbalanced problem by
more: about 6.5 points of balanced accuracy and 6.2 of macro-F1.

The last two rows locate the cause. Lesions photographed only once cannot leak under either
protocol, and on them the two splits differ by 0.3 points on average. The gap comes from lesions
photographed more than once: under the naive split an average of 535 of the 1,503 test images
(36%) have another image of the same lesion in the training set, and accuracy on multi-image
lesions rises by 5.4 points.

The naive numbers do not describe a better model. They describe the same model graded partly on
lesions it has already seen. Every other result in this repo uses the lesion-grouped split.

Seeds are fixed, and a rerun on the same machine reproduced both result files byte for byte.

Per-seed results are in [`reports/leakage_demo_runs.csv`](reports/leakage_demo_runs.csv) and the
summary above is [`reports/leakage_demo.csv`](reports/leakage_demo.csv). Reproduce both with
`make leakage-demo`.

## The text modality

Each image's metadata is turned into one templated sentence, for example *"65-year-old male with a
skin lesion on the face, imaged by dermatoscopy."* The template, its phrase tables, and examples
for every class are in [`reports/text_examples.md`](reports/text_examples.md), written by
`scripts/text_examples.py`. It is synthetic text, not clinical notes.

HAM10000 also records how each diagnosis was established (histopathology, follow-up, expert
consensus, or confocal microscopy). That field is **left out of the text** because it leaks the
label: every follow-up image is a nevus, every confocal image is a benign keratosis, and every
melanoma was confirmed by histopathology. It was used once, in the ablation below, to measure the
size of the leak.

## Unimodal baselines

Both baselines are trained by the same code under identical conditions: the lesion-grouped splits,
AdamW with warmup and cosine decay, 20 epochs, a class-weighted loss, and the checkpoint from the
epoch with the best validation macro-F1 (details in [`DECISIONS.md`](DECISIONS.md)). Each is
evaluated once on the test split. Mean ± sd over three seeds:

| Model                                               |        Macro-F1 | Balanced accuracy | Melanoma recall |
| --------------------------------------------------- | --------------: | ----------------: | --------------: |
| Image only (EfficientNet-B0)                        | 0.7181 ± 0.0141 |   0.7199 ± 0.0143 | 0.7341 ± 0.0150 |
| Text only (DistilBERT)                              | 0.2262 ± 0.0162 |   0.3164 ± 0.0296 | 0.2083 ± 0.0449 |
| Text only + diagnosis method (leak, not a baseline) | 0.3075 ± 0.0084 |   0.3985 ± 0.0156 | 0.3313 ± 0.0658 |

| Class recall                                            |      Image only |       Text only | Text + diagnosis method |
| ------------------------------------------------------- | --------------: | --------------: | ----------------------: |
| actinic keratosis / intraepithelial carcinoma (`akiec`) | 0.7255 ± 0.0707 | 0.1961 ± 0.2066 |         0.5425 ± 0.1822 |
| basal cell carcinoma (`bcc`)                            | 0.7835 ± 0.0270 | 0.2165 ± 0.1081 |         0.4199 ± 0.0600 |
| benign keratosis-like lesion (`bkl`)                    | 0.6964 ± 0.0321 | 0.3163 ± 0.0327 |         0.1826 ± 0.0205 |
| dermatofibroma (`df`)                                   | 0.5758 ± 0.1144 | 0.6818 ± 0.0909 |         0.3182 ± 0.0909 |
| melanoma (`mel`)                                        | 0.7341 ± 0.0150 | 0.2083 ± 0.0449 |         0.3313 ± 0.0658 |
| melanocytic nevus (`nv`)                                | 0.8873 ± 0.0098 | 0.4897 ± 0.0386 |         0.6613 ± 0.0405 |
| vascular lesion (`vasc`)                                | 0.6364 ± 0.0909 | 0.1061 ± 0.0262 |         0.3333 ± 0.0946 |

- **The image carries the signal.** The image-only model reaches a macro-F1 of 0.718, yet still
  misses roughly a quarter of melanomas (recall 0.734).
- **Metadata without the leak is only weakly informative.** Text-only balanced accuracy is 0.316,
  above the 0.143 of chance with seven classes but far below the image. That is a property of the
  text, not an undertrained model: 91% of images share their exact description with an image of a
  different diagnosis, and a lookup table that memorises each training description's class
  frequencies reaches a balanced accuracy of 0.305, which DistilBERT slightly exceeds
  ([`reports/text_examples.md`](reports/text_examples.md)).
- **The diagnosis method is a leak, and the ablation shows it.** Adding that one sentence lifts
  text-only balanced accuracy from 0.316 to 0.399 and macro-F1 from 0.226 to 0.308. The sentence
  splits the data along the diagnostic workup: follow-up examination only ever appears for nevi,
  and every melanoma, basal cell carcinoma, and actinic keratosis is histopathology-confirmed. That
  row measures the leak; it is not a baseline.
- **The bar for fusion is the image-only model.** Fusion has to beat a macro-F1 of 0.718 ± 0.014 by
  more than seed noise for the text branch to earn its place.

Per-seed results are in [`reports/baselines_runs.csv`](reports/baselines_runs.csv), the summary is
[`reports/baselines.csv`](reports/baselines.csv), and every run, with its per-epoch validation
curve, is logged to the local MLflow store in `mlruns/`. Reproduce with `make train`.

## Setup

Requires Python 3.11 and an NVIDIA GPU with a CUDA 12 capable driver.

```
make setup          # creates .venv and installs pinned dependencies with CUDA 12.6 PyTorch wheels
make lint
make test
make data           # writes configs/splits.json, builds the image cache, writes dataset statistics
make leakage-demo   # grouped versus naive split comparison
make train          # image-only and text-only baselines, three seeds each
```

`make setup` calls `python3.11` (`py -3.11` on Windows). Point it at a different interpreter with
`make setup PYTHON=/path/to/python3.11`.

`make data` decodes and resizes every image once on the GPU and caches the result under
`data/processed/` (about 1.5 GB). Training then runs from GPU memory with no data-loading workers,
which keeps CPU load low. `make lint` and `make test` need neither the dataset nor a GPU.

| Target              | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `make setup`        | create the virtual environment and install dependencies               |
| `make test`         | run the test suite                                                    |
| `make lint`         | ruff lint and format check                                            |
| `make format`       | apply ruff formatting and autofixes                                   |
| `make data`         | lesion-grouped splits, image cache, dataset statistics, text examples |
| `make leakage-demo` | train under grouped and naive splits and compare                      |
| `make train`        | train and evaluate the unimodal baselines                             |
| `make eval`         | classification, faithfulness, and sanity checks (not implemented yet) |
| `make report`       | assemble `reports/` (not implemented yet)                             |

Browse logged runs with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

All settings live in [`configs/default.yaml`](configs/default.yaml). Non-obvious choices are
logged with their reasons in [`DECISIONS.md`](DECISIONS.md).

## License

[MIT](LICENSE)
