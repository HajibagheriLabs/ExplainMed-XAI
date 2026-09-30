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

## Headline metrics are macro-F1, balanced accuracy, and per-class recall

Melanocytic nevi dominate the class distribution, so a model that always predicts nevus scores a
high accuracy while being useless. Accuracy is never reported as a headline number. Melanoma recall
is called out separately because missing a melanoma is the costly error.

## GPU-first, CPU-light compute

The reference workstation pairs a 24 GB GPU with a 4-core CPU, so the CPU is the bottleneck.
Training, evaluation, and attribution run on CUDA (`device: cuda`), and `cpu_threads` caps PyTorch's
intra-op CPU threads. CPU-bound work such as JPEG decoding and augmentation is kept off the hot path
where possible.

## PyTorch 2.14 with CUDA 12.6 wheels

CUDA 12.6 is the newest CUDA 12 wheel index that carries the latest PyTorch release. PyPI only
serves CPU-only PyTorch wheels on Windows, so `make setup` passes the PyTorch index as an extra
index; pip then prefers the `+cu126` build because a local version label sorts above the bare one.

## Strict config loading

The YAML config is loaded into frozen dataclasses. Missing keys, unknown keys, and type mismatches
raise instead of defaulting. This catches typos in key names and the PyYAML quirk where `1e-4`
(without a decimal point) is read as a string.
