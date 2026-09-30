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

Data layer, unimodal baselines, and the fusion model are done; the explanation stages are not.
Test-split results, mean ± sd over three seeds:

| Model      |        Macro-F1 | Balanced accuracy | Melanoma recall |
| ---------- | --------------: | ----------------: | --------------: |
| Image only | 0.7124 ± 0.0235 |   0.7096 ± 0.0205 | 0.6905 ± 0.0273 |
| Text only  | 0.2266 ± 0.0061 |   0.3206 ± 0.0085 | 0.1687 ± 0.0656 |
| Fusion     | 0.7216 ± 0.0111 |   0.7146 ± 0.0062 | 0.6885 ± 0.0034 |

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
| Image only (EfficientNet-B0)                        | 0.7124 ± 0.0235 |   0.7096 ± 0.0205 | 0.6905 ± 0.0273 |
| Text only (DistilBERT)                              | 0.2266 ± 0.0061 |   0.3206 ± 0.0085 | 0.1687 ± 0.0656 |
| Text only + diagnosis method (leak, not a baseline) | 0.3163 ± 0.0194 |   0.4113 ± 0.0269 | 0.4187 ± 0.1899 |

| Class recall                                            |      Image only |       Text only | Text + diagnosis method |
| ------------------------------------------------------- | --------------: | --------------: | ----------------------: |
| actinic keratosis / intraepithelial carcinoma (`akiec`) | 0.7582 ± 0.0566 | 0.3660 ± 0.0967 |         0.4510 ± 0.2121 |
| basal cell carcinoma (`bcc`)                            | 0.7619 ± 0.0825 | 0.2035 ± 0.0940 |         0.3030 ± 0.1261 |
| benign keratosis-like lesion (`bkl`)                    | 0.6921 ± 0.0224 | 0.2845 ± 0.1237 |         0.1762 ± 0.0287 |
| dermatofibroma (`df`)                                   | 0.5455 ± 0.0909 | 0.6515 ± 0.1144 |         0.4394 ± 0.0525 |
| melanoma (`mel`)                                        | 0.6905 ± 0.0273 | 0.1687 ± 0.0656 |         0.4187 ± 0.1899 |
| melanocytic nevus (`nv`)                                | 0.8977 ± 0.0095 | 0.4640 ± 0.0600 |         0.6970 ± 0.0730 |
| vascular lesion (`vasc`)                                | 0.6212 ± 0.0525 | 0.1061 ± 0.0525 |         0.3939 ± 0.0525 |

- **The image carries the signal.** The image-only model reaches a macro-F1 of 0.712, yet still
  misses about three in ten melanomas (recall 0.691).
- **Metadata without the leak is only weakly informative.** Text-only balanced accuracy is 0.321,
  above the 0.143 of chance with seven classes but far below the image. That is a property of the
  text, not an undertrained model: 91% of images share their exact description with an image of a
  different diagnosis, and a lookup table that memorises each training description's class
  frequencies reaches a balanced accuracy of 0.305, which DistilBERT slightly exceeds
  ([`reports/text_examples.md`](reports/text_examples.md)).
- **The diagnosis method is a leak, and the ablation shows it.** Adding that one sentence lifts
  text-only balanced accuracy from 0.321 to 0.411 and macro-F1 from 0.227 to 0.316. The sentence
  splits the data along the diagnostic workup: follow-up examination only ever appears for nevi,
  and every melanoma, basal cell carcinoma, and actinic keratosis is histopathology-confirmed. That
  row measures the leak; it is not a baseline.
- **The bar for fusion is the image-only model.** Fusion has to beat a macro-F1 of 0.712 ± 0.024 by
  more than seed noise for the text branch to earn its place.

Per-seed results are in [`reports/baselines_runs.csv`](reports/baselines_runs.csv) and the summary
is [`reports/baselines.csv`](reports/baselines.csv).

## Fusion

The fusion model concatenates the image features (EfficientNet-B0, 1280-d) and the text features
(DistilBERT [CLS], 768-d) and classifies them with a small MLP. Both encoders start from the same
pretrained weights as the baselines and are trained end to end by the same code, with the same
seeds, epochs, and learning rates. The text excludes the diagnosis method.

| Model                                      |         Macro-F1 | Balanced accuracy |  Melanoma recall |
| ------------------------------------------ | ---------------: | ----------------: | ---------------: |
| Image only                                 |  0.7124 ± 0.0235 |   0.7096 ± 0.0205 |  0.6905 ± 0.0273 |
| Text only                                  |  0.2266 ± 0.0061 |   0.3206 ± 0.0085 |  0.1687 ± 0.0656 |
| **Fusion**                                 |  0.7216 ± 0.0111 |   0.7146 ± 0.0062 |  0.6885 ± 0.0034 |
| Fusion − image only (paired by seed)       | +0.0092 ± 0.0303 |  +0.0050 ± 0.0244 | -0.0020 ± 0.0300 |
| Fusion, text shuffled across test images   |  0.6674 ± 0.0245 |   0.6573 ± 0.0216 |  0.6845 ± 0.0372 |
| Fusion, images shuffled across test images |  0.1439 ± 0.0055 |   0.1447 ± 0.0052 |  0.1270 ± 0.0241 |

**Fusion adds little over the image alone.** Paired by seed, the macro-F1 gain is +0.009 ± 0.030
(+0.032, +0.021, and −0.025 across the three seeds) and the balanced-accuracy gain is +0.005. Both
are smaller than the seed-to-seed variation of the image-only model, so this experiment gives no
evidence that the templated metadata improves classification.

**Fusion does use the text, but it does not help.** Given the text of a different test image,
fusion drops to a macro-F1 of 0.667, below the image-only model, so its predictions do depend on the
text. Given the correct text, it only matches the image-only model. Shuffling the images instead
drops fusion to chance (balanced accuracy 0.145 against 1/7): the text cannot carry the prediction
on its own. The fusion model has learned to lean on a weak signal that, on this data, buys nothing
the image did not already provide.

### Melanoma recall

Melanoma is the class where a miss matters most, and neither model is good at it. Recall is
0.691 ± 0.027 for image only and 0.689 ± 0.003 for fusion: about three in ten melanomas in the test
split are missed, and adding the metadata changes nothing. The confusion matrices show where they
go. Pooled over the three fusion seeds (504 melanoma predictions), 20% of melanomas
are called melanocytic nevi (19% for image only), the most dangerous error because it
labels a malignant lesion as a common benign mole, and 6% are called benign keratosis.
In the other direction, 6% of nevi are called melanoma.

![Confusion matrices of the image-only and fusion models, rows normalised by true class](reports/figures/confusion_matrices.png)

The counts behind the figure are in [`reports/confusion_matrices.csv`](reports/confusion_matrices.csv),
per-seed results in [`reports/fusion_runs.csv`](reports/fusion_runs.csv), and the summary above is
[`reports/fusion_results.csv`](reports/fusion_results.csv). Every training run and its validation
curve is logged to the local MLflow store in `mlruns/`. Reproduce the baselines and fusion with
`make train`; a rerun of the fusion training on the same machine reproduced its per-seed results
and confusion counts byte for byte.

## Setup

Requires Python 3.11 and an NVIDIA GPU with a CUDA 12 capable driver.

```
make setup          # creates .venv and installs pinned dependencies with CUDA 12.6 PyTorch wheels
make lint
make test
make data           # writes configs/splits.json, builds the image cache, writes dataset statistics
make leakage-demo   # grouped versus naive split comparison
make train          # baselines and fusion model, three seeds each
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
| `make train`        | train and evaluate the baselines and the fusion model                 |
| `make eval`         | classification, faithfulness, and sanity checks (not implemented yet) |
| `make report`       | assemble `reports/` (not implemented yet)                             |

Browse logged runs with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

All settings live in [`configs/default.yaml`](configs/default.yaml). Non-obvious choices are
logged with their reasons in [`DECISIONS.md`](DECISIONS.md).

## License

[MIT](LICENSE)
