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
