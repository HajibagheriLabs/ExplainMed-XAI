# Model card: ExplainMed fusion classifier

> **Not a diagnostic tool.** This model is a research artefact for studying whether saliency
> explanations can be trusted. It has not been clinically validated, must not be used to make or
> support decisions about any patient, and nothing it outputs is a diagnosis.

## Model details

- **Architecture.** Late fusion of an image encoder and a text encoder. The image encoder is an
  EfficientNet-B0 trunk pretrained on ImageNet (1280-d pooled features). The text encoder is
  `distilbert-base-uncased`, pooled at the [CLS] token (768-d). The two feature vectors are
  concatenated and classified into seven diagnostic categories by an MLP (dropout, 256 GELU units,
  dropout, linear).
- **Training.** End to end with AdamW, linear warmup then cosine decay, 20 epochs, batch 128,
  fp16 autocast, inverse-frequency class weights, and the checkpoint from the epoch with the best
  validation macro-F1. Three models were trained, under seeds 42, 43, and 44. All settings are in
  [`configs/default.yaml`](../configs/default.yaml) and the reasons in
  [`DECISIONS.md`](../DECISIONS.md).
- **Explanations.** Grad-CAM and integrated gradients on the image branch, and attention rollout
  on the text branch ([`src/explainmed/explain.py`](../src/explainmed/explain.py)).
- **Distribution.** Trained weights are not distributed. `make train` reproduces them.

## Inputs

- A dermatoscopic image, squashed from the full frame to 224×224 pixels.
- **A synthetic sentence, not a clinical note.** HAM10000 has no free text. The text input is built
  by a fixed template from three structured metadata fields: age, sex, and body site. For example,
  *"45-year-old male with a skin lesion on the back, imaged by dermatoscopy."* Real clinical text
  is longer, noisier, and carries information this template never does. Nothing measured here
  says how the model would behave on real notes. The template is documented in
  [`text_examples.md`](text_examples.md). The field recording how each diagnosis was established
  is excluded because it leaks the label.

## Intended use

- Research on how to evaluate explanations of image and multimodal classifiers: faithfulness
  metrics, sanity checks, and the failure cases they reveal.
- Teaching and demonstrating why lesion-grouped splits, class-balanced metrics, and measured
  (not illustrated) explanations matter in medical machine learning.

## Out-of-scope use

- Any clinical use: diagnosis, triage, screening, second opinions, patient-facing tools, or
  prioritising which lesions a clinician examines.
- Using an explanation map as a justification for a decision, or as evidence that a prediction is
  correct. This repo shows that the maps do not carry that information
  ([`misleading_explanations.md`](misleading_explanations.md)).
- Clinical photographs, images from other dermatoscopes or clinics, or populations unlike the
  training data (see below), without a new evaluation.
- Real clinical text in place of the templated sentence.
- Lesion types outside the seven HAM10000 categories. The model always returns one of the seven.

## Training data

[HAM10000](https://doi.org/10.7910/DVN/DBW86T): 10,015 dermatoscopic images of 7,470 lesions,
collected at the Department of Dermatology of the Medical University of Vienna and at a skin cancer
practice in Queensland, Australia. The split is grouped by lesion, stratified by diagnosis, and
committed in [`configs/splits.json`](../configs/splits.json): 7,054 training, 1,464 validation, and
1,497 test images.

Limitations of the data that carry over to the model
([`dataset_stats.csv`](dataset_stats.csv)):

- **Skin tone.** HAM10000 does not record skin type, and its images come from clinics serving
  predominantly fair-skinned populations. Work estimating skin tone from the images of the ISIC
  challenge data, which include HAM10000, found them to be overwhelmingly light (Kinyanjui et al.,
  2020). Lesions on darker skin look different under dermatoscopy, and melanoma on darker skin is
  more often acral. **This model has effectively never seen dark skin, and no result in this repo
  says anything about how it performs on it.** This is a documented fairness limitation of the
  dataset.
- **Class balance.** 67% of the images are melanocytic nevi. Dermatofibroma (1.1%) and vascular
  lesions (1.4%) have the fewest examples and the lowest recall.
- **Age and sex.** 54% of images are of male patients and 45% of female patients. Three quarters
  come from patients aged 35 to 70, and fewer than 3% from patients under 20, so children and
  adolescents are barely represented.
- **Selection.** Every melanoma, basal cell carcinoma, and actinic keratosis in the dataset was
  confirmed by histopathology ([`text_examples.md`](text_examples.md)), so the malignant lesions are
  ones suspicious enough to have been excised. The class frequencies are not those of any screening
  population.

## Evaluation protocol

- The test split is evaluated once per trained model, and results are reported as mean ± sd over
  the three seeds.
- Headline metrics are macro-F1, balanced accuracy, and per-class recall. Accuracy is not a
  headline, because a model that always predicts nevus is 67% accurate.
- The explanation methods are evaluated with deletion and insertion AUC against a random-attribution
  control, sparsity, localisation against the lesion segmentation, inter-method agreement, and the
  Adebayo et al. model- and label-randomisation checks ([`explanation_analysis.md`](explanation_analysis.md)).

## Results

Test split, mean ± sd over three seeds ([`results.csv`](results.csv)):

| Metric            |      Image only |          Fusion |
| ----------------- | --------------: | --------------: |
| Macro-F1          | 0.7124 ± 0.0235 | 0.7216 ± 0.0111 |
| Balanced accuracy | 0.7096 ± 0.0205 | 0.7146 ± 0.0062 |
| Recall, `akiec`   | 0.7582 ± 0.0566 | 0.7908 ± 0.0113 |
| Recall, `bcc`     | 0.7619 ± 0.0825 | 0.7316 ± 0.0641 |
| Recall, `bkl`     | 0.6921 ± 0.0224 | 0.6943 ± 0.0169 |
| Recall, `df`      | 0.5455 ± 0.0909 | 0.5606 ± 0.0694 |
| Recall, `mel`     | 0.6905 ± 0.0273 | 0.6885 ± 0.0034 |
| Recall, `nv`      | 0.8977 ± 0.0095 | 0.9003 ± 0.0150 |
| Recall, `vasc`    | 0.6212 ± 0.0525 | 0.6364 ± 0.0909 |

Fusion's gain over the image-only model is within seed noise (macro-F1 +0.0092 ± 0.0303, paired by
seed). The synthetic text does not measurably help.

## Known failure modes

- **Missed melanoma.** About three in ten test melanomas are missed. Pooled over the three fusion
  models, 20% of melanomas are called melanocytic nevi, the most dangerous error, and 6% benign
  keratosis ([`confusion_matrices.csv`](confusion_matrices.csv)).
- **Rare classes.** Dermatofibroma recall is 0.56 and vascular-lesion recall 0.64.
- **Confident errors.** Between 87 and 99 of the 1,497 test predictions per model are wrong with a
  probability of at least 0.9. Between 24 and 30 of them are malignant lesions called benign
  ([`misleading_counts.csv`](misleading_counts.csv)).
- **Explanations that look right on wrong predictions.** About 47% of confident errors come with a
  Grad-CAM map as well placed on the lesion as that of a typical correct prediction, including 13 to
  17 missed malignancies per model. How well the map sits on the lesion separates wrong from correct
  predictions at an AUROC of 0.53, against 0.85 for the model's own confidence
  ([`misleading_explanations.md`](misleading_explanations.md)).
- **Unreliable explanation methods.** No attribution method passes every check. Attention rollout
  fails both sanity checks, integrated gradients fails model randomisation, and Grad-CAM fails label
  randomisation narrowly ([`explanation_analysis.md`](explanation_analysis.md)).
- **Distribution shift.** Every number here comes from the same two clinics, their dermatoscopes,
  and their patients. Performance on other devices, image acquisition protocols, or populations is
  unknown.

## Ethical considerations

The failure modes above are the reason the model must not be used on patients. A plausible
explanation next to a confident prediction would invite trust that this repo shows is unwarranted,
and the missing skin-tone coverage means any error rate measured here could be worse, by an unknown
amount, for patients with darker skin.
