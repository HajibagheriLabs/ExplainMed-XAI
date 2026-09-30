# Decisions

A short log of every non-obvious choice and why it was made.

## HAM10000 rather than MIMIC-CXR

MIMIC-CXR needs PhysioNet credentialing and a training course, so a reader could not reproduce the
results. HAM10000 is openly downloadable, has image-level metadata to build a second modality from,
and is large enough to train on.

## The text modality is synthetic

HAM10000 has no clinical notes. The text branch sees sentences templated from structured metadata
(age, sex, localisation, diagnosis method). This is a legitimate way to study multimodal fusion and
text-side attribution, but it is not clinical text, and this is stated in the README, the model
card, and here. Nothing in the repo describes it as clinical notes.

## Splits are grouped by lesion, not by image

Several images often show the same lesion. Splitting by image would put near-duplicates of a test
lesion into training and inflate every metric. Every split is grouped by `lesion_id`, and a test
asserts there is zero `lesion_id` overlap across splits.

The split is drawn over lesions, not images: each lesion has exactly one diagnosis (checked when the
metadata is loaded), so a two-stage stratified shuffle over the lesion table gives test, then
validation, then train. Stratification is therefore exact in lesions and only approximate in images,
because the number of images per lesion differs between classes. `reports/dataset_stats.csv` reports
both.

`configs/splits.json` stores lesion ids, not image ids, together with the seed and fractions that
produced it. Storing lesions makes the grouping unit explicit, and loading raises if the file was
generated under a different seed or fractions than the active config.

## Headline metrics are macro-F1, balanced accuracy, and per-class recall

Melanocytic nevi dominate the class distribution, so a model that always predicts nevus scores a
high accuracy while being useless. Accuracy is never reported as a headline number. Melanoma recall
is called out separately because missing a melanoma is the costly error.

## GPU-first, CPU-light compute

The reference workstation pairs a 24 GB GPU with a 4-core CPU, so the CPU is the bottleneck.
Training, evaluation, and attribution run on CUDA (`device: cuda`), and `cpu_threads` caps PyTorch's
intra-op CPU threads. CPU-bound work such as JPEG decoding and augmentation is kept off the hot path
where possible.

## Images are decoded once on the GPU and cached at model resolution

`make data` decodes every JPEG with nvjpeg on the GPU, resizes it there, and stores the result as one
uint8 tensor under `data/processed/`. Training then holds the whole dataset in GPU memory (about
1.5 GB at 224 px) and indexes batches directly, with no DataLoader workers and no per-epoch decoding.

The full 600x450 frame is squashed to a square rather than centre-cropped. Cropping would cut lesions
that reach the image border, and keeping the whole frame keeps the lesion segmentation masks aligned
with the input for the localisation metric. The cost is a fixed 4:3 to 1:1 distortion, identical in
training and evaluation.

GPU and CPU JPEG decoders differ by a few intensity levels per pixel, so a cache built with
`device: cpu` is not bit-identical to one built on the GPU.

## Augmentation is limited to flips and transposes

Dermatoscopic images have no canonical orientation, so the eight symmetries of the square are
label-preserving. They need no interpolation, run on the GPU, and leave pixel values untouched. The
evaluation transform is normalisation only and is therefore deterministic.

## Default memory layout, not channels_last

With this PyTorch and cuDNN build, mixed-precision training of a ResNet-18 in channels_last ran
several times slower than in the default layout on the same GPU, while inference was unaffected.
Training therefore uses the default layout.

## PyTorch 2.14 with CUDA 12.6 wheels

CUDA 12.6 is the newest CUDA 12 wheel index that carries the latest PyTorch release. PyPI only
serves CPU-only PyTorch wheels on Windows, so `make setup` passes the PyTorch index as an extra
index; pip then prefers the `+cu126` build because a local version label sorts above the bare one.

## Strict config loading

The YAML config is loaded into frozen dataclasses. Missing keys, unknown keys, and type mismatches
raise instead of defaulting. This catches typos in key names and the PyYAML quirk where `1e-4`
(without a decimal point) is read as a string.

## Leakage demo: paired repeats with a control group

A single pair of runs cannot separate leakage from sampling noise. `scripts/leakage_demo.py` therefore
redraws both the lesion-grouped and the naive image-level split under five seeds, retrains the same
model on each, and reports the naive-minus-grouped gap as a paired difference per seed. The first
repeat uses the config seed, so its grouped split is the committed `configs/splits.json`.

Accuracy is also reported separately for single-image lesions, which cannot leak under either split
and act as the control, and for multi-image lesions, which can. Comparing leaked and non-leaked test
images inside the naive split alone would be confounded: `reports/dataset_stats.csv` shows that
melanoma and the other non-nevus classes have more images per lesion than nevi, so multi-image
lesions are a harder mix of classes.

The model is an ImageNet ResNet-18 fine-tuned with hyperparameters fixed in the config before any
result was seen. There is no model selection, so the validation split is unused and both protocols
train on their training split only.

## Text template: explicit phrase tables, one fixed sentence

`src/explainmed/text.py` maps each metadata value to a phrase through lookup tables and assembles
one sentence per image (`<patient> with a skin lesion <site>, imaged by dermatoscopy.`). There is no
paraphrasing or random variation, so a reader can reconstruct exactly what the text branch sees,
and an unexpected metadata value raises instead of producing an unseen phrase.
`reports/text_examples.md` lists the tables and examples.

The dataset-source column (`dataset`) is not templated either: which clinic an image came from is
not a property of the lesion, and some sources are dominated by nevi.

## Diagnosis method is excluded from the main text modality

The playbook asked for age, sex, localisation, and diagnosis method. The first three are templated.
The diagnosis method (`dx_type`) is not part of the main text modality because it is target
leakage: it records how the ground truth was established, which follows from the diagnosis
(suspicious lesions are excised for histopathology, benign-looking ones are followed up).
In HAM10000, every follow-up image is a nevus, every confocal image is a benign keratosis, and
every melanoma, basal cell carcinoma, and actinic keratosis is histopathology-confirmed
(`reports/text_examples.md`). A model given this sentence would be partly reading the label.

The template still supports it, and the text-only baseline is trained both ways. The row with the
diagnosis method is reported as a measurement of the leak, not as a baseline to beat, and the
fusion model uses the text without it.

## Encoders

The image encoder is an ImageNet EfficientNet-B0 trunk (global-average-pooled 1280-d features). It
is small enough to fine-tune many times on one GPU, and a convolutional trunk gives Grad-CAM a
natural target layer. The text encoder is `distilbert-base-uncased`, pooled at the [CLS] token.
Each encoder plus a dropout and linear head is a standalone classifier, and the fusion model will
reuse the same encoder classes.

## Identical training conditions for every model

Every model is trained by the same `fit` function: the committed lesion-grouped splits, AdamW,
linear warmup then cosine decay, 20 epochs, batch 128, fp16 autocast, inverse-frequency class
weights in the cross-entropy, and the checkpoint from the epoch with the best validation macro-F1.
The test split is evaluated once per run. Each model is trained under three seeds and reported as
mean and standard deviation, so that differences between models can be compared with seed noise.

The only per-model setting is the learning rate of each pretrained encoder (EfficientNet 3e-4,
DistilBERT 3e-5, heads 1e-3). These are the standard fine-tuning rates for each architecture and
apply to that encoder in every model that contains it, including fusion. They were fixed before
training and not tuned.

Class weights are used because the headline metrics weight every class equally. Batch 128 keeps
the GPU, not the single CPU thread that dispatches work to it, as the bottleneck. fp16 was chosen
over bf16 because it trained EfficientNet-B0 faster on the reference GPU.

## MLflow uses a SQLite store inside `mlruns/`

MLflow 3.16 refuses the plain file store (`./mlruns`) unless an environment override is set. The
tracking database is therefore `mlruns/mlflow.db` with artifacts under `mlruns/artifacts/`, which
keeps every run in the gitignored `mlruns/` directory. Browse it with
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## Fusion: late fusion by concatenation and a small MLP

The fusion model concatenates the pooled EfficientNet-B0 features (1280-d) and the DistilBERT [CLS]
state (768-d) and passes them through dropout, a 256-unit GELU layer, and a linear classifier. Late
fusion keeps each encoder identical to its unimodal baseline, so any difference in results comes
from combining them. The hidden layer lets the classifier model interactions between modalities
(an appearance that means something different at a different age or body site), which a single
linear layer over the concatenation could not.

Both encoders start from the same pretrained weights as the baselines and are trained end to end by
the same `fit` function with the same seeds, epochs, and per-encoder learning rates. Starting from
the trained unimodal checkpoints instead would give fusion extra training and break the comparison.
The text is the version without the diagnosis method.

## Fusion gain is a paired difference, and modality use is measured directly

Fusion and image-only runs share seeds, and therefore batch order and augmentation, so the gain is
computed per seed and summarised as a paired difference.

A higher score does not show that the model uses the text. After training, each fusion checkpoint is
re-evaluated with the text inputs permuted across test images, and again with the images permuted.
Permuting a modality breaks its link to the label while keeping its distribution, so the drop in
performance measures how much the model relies on that modality.

## Every run is seeded before its model is built

The classifier heads are randomly initialised when the model is constructed. The first version of
the baseline script seeded only inside `fit`, after construction, so head initialisation depended
on the RNG state left by earlier runs, and a single run could not be reproduced on its own.
`train_and_evaluate` now seeds before building each model. The baselines were retrained after this
fix, so their committed results come from the current code.

## Attribution methods share one output convention

Every method explains the class the model predicted, because that is the decision a reader would be
shown an explanation for. Each returns non-negative evidence for that class over its modality's
input grid, scaled to a maximum of 1: pixels for the image methods and tokens for the text method.
Negative evidence is dropped (Grad-CAM's ReLU, the positive part of integrated gradients), so every
map answers the same question. Faithfulness metrics and sanity checks then treat all methods alike,
and the only modality-specific code is how a pixel or a token is removed.

Grad-CAM targets the last convolutional block of EfficientNet-B0, the deepest layer that still has
spatial resolution (7×7). Integrated gradients uses 64 steps from a baseline of zeros after
normalisation, which is the ImageNet mean colour. A black baseline would scale each pixel's
attribution by its distance from black, giving the darkest structures, pigment networks and
blotches, the least attribution. Attention rollout averages heads, adds the identity for the
residual connection, and multiplies through the layers (Abnar and Zuidema, 2020). It does not depend
on the class being explained, and the sanity checks test what that implies.

Grad-CAM and integrated gradients come from Captum, which runs batched on the GPU against the fusion
model with the text held fixed. The `grad-cam` package was dropped because it resizes every map
with OpenCV on the CPU.

In the example figures, image maps saturate at their 99th percentile, because integrated gradients
concentrates on a few pixels and would otherwise be invisible. Word pieces are merged into words
for display, each word taking its strongest piece's weight. Both are display choices only; every
metric uses the raw maps.

## Faithfulness is measured on patches, against a random control

Deletion and insertion rank units by attribution and remove, or restore, them in 5% steps while
tracking the probability of the predicted class. The area under each curve summarises it. Image
units are 8×8-pixel patches, a 28×28 grid, because a pixel ranking would compare an upsampled 7×7
Grad-CAM map with a speckled integrated-gradients map on pixel noise rather than on regions.
Sparsity (Gini index), localisation, and agreement use the same grid. Text units are content
tokens, and a removed token is masked out of attention, which removes it without shifting the
positions of the others.

A removed patch is replaced by the same image blurred with a sigma of 8 pixels. A constant fill
would insert sharp edges, a strong signal the model never saw in training, while blur removes the
local texture a diagnosis depends on and keeps the colour layout. The reference, the patch size,
and the 20 steps were fixed before any result was seen. Deletion and insertion are known to depend
on the reference, so the image methods are scored again with a mean-colour reference as a
sensitivity check; the verdicts use the blurred reference.

The control is a uniform random attribution over the same units and images. Each method is
compared with it as a paired difference per image, with a 95% normal-approximation interval over
the 500 images. A method is better or worse than random on a curve only if the interval excludes
zero on that side in all three seeds; otherwise the comparison is inconclusive. Ties in a ranking,
such as the zero region of a Grad-CAM map, are broken at random rather than in raster order.

## Sanity checks follow each method's branch, with a threshold fixed in advance

Cascading randomisation follows the path a method's gradients or attention take: the fusion
MLP's output layer, then its hidden layer, then either the nine EfficientNet blocks from the top
down (Grad-CAM and integrated gradients) or the six DistilBERT layers and the embeddings
(attention rollout). Each layer is re-initialised with PyTorch's default initialiser for every
parameterised submodule, batch-norm statistics included, and the code raises if any parameter
would be missed. At every step the explained class is the trained model's prediction, so only the
weights change.

Similarity to the original map is Spearman rank correlation on the patch or token grid, with ties
given their mean rank, reported alongside top-10% overlap. A map that randomisation flattens to a
constant counts as similarity 0. A method passes a check only if its mean rank correlation stays
below 0.5 at every cascade step and in every seed. The threshold was written into the config
before either check was run. A map that survives any stage does not depend on the weights that
stage destroyed.

Label randomisation retrains the fusion model from the same pretrained weights for 10 epochs on
labels permuted across all images, which keeps the class balance and breaks the link between image
and label. The checkpoint is the epoch with the best macro-F1 against the shuffled training labels,
the one that memorised them best, so no true label influences it. As a reference, the same
similarity is measured between two fusion models trained on true labels under different seeds:
that is how alike explanations are when two models learned the same task.
