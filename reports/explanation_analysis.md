# Explanation analysis

Each attribution method gets a verdict, with the evidence behind it. All numbers below come from
`scripts/evaluate_faithfulness.py` and `scripts/sanity_checks.py` (`make eval`). They are means over
the three fusion models (seeds 42, 43, 44), computed on a fixed, seeded sample of the lesion-grouped
test split: 500 images for faithfulness, and the first 200 of them for the sanity checks. The
protocol and every threshold were fixed before the first run and are described in
[`DECISIONS.md`](../DECISIONS.md).

## Verdicts

| Method               | Deletion vs random | Insertion vs random | Model randomisation         | Label randomisation         | Verdict   |
| -------------------- | ------------------ | ------------------- | --------------------------- | --------------------------- | --------- |
| Grad-CAM             | better             | **worse**           | passes (highest 0.46)       | **fails** (0.50; seed 44 0.57) | **fails** |
| Integrated gradients | better             | **worse**           | **fails** (0.72 at step 1)  | passes (0.37)               | **fails** |
| Attention rollout    | better (tiny)      | better (tiny)       | **fails** (1.00 at step 1)  | **fails** (0.89)            | **fails** |

"Better" or "worse" means the paired difference from the random control has a 95% interval that
excludes zero on that side in all three seeds. The randomisation columns give the mean rank
correlation between the original map and the map after randomisation; a check fails when that
correlation is 0.5 or higher, at any step and in any seed.

**No method passes every check.** Grad-CAM comes closest and is the only one whose failure is
narrow. Integrated gradients fails the model-randomisation test decisively, and attention rollout
fails both tests. Presenting the three maps side by side as equally valid explanations would be
wrong.

## Grad-CAM: finds a region the model needs, but not only for the reason it predicts

**Faithfulness: mixed.** Deleting the patches Grad-CAM ranks highest destroys the prediction
fastest. The deletion AUC is 0.591 against 0.688 for random attribution, a paired difference of
−0.100, better than random in every seed. The result holds with a mean-colour reference in place of
the blur (−0.098, [`faithfulness_reference_check.csv`](faithfulness_reference_check.csv)).

Insertion goes the other way. Restoring the Grad-CAM patches first into a blurred image recovers the
prediction more slowly than restoring random patches: 0.605 against 0.689, a difference of −0.074,
worse than random in every seed. With the mean-colour reference the difference shrinks to −0.032 and
is no longer consistent across seeds. The curves in
[`figures/faithfulness_curves.png`](figures/faithfulness_curves.png) show why. Removing the top 15%
of Grad-CAM patches drops the predicted-class probability from 0.92 to 0.54, close to the 0.49 of a
fully blurred image. Restoring those same patches alone leaves it near 0.5 until more than half the
image is back. The highlighted region is necessary for the prediction but not sufficient. The model
also uses the surrounding skin and the lesion border, which Grad-CAM ranks low.

**Localisation and sparsity: strong.** On average, 69% of Grad-CAM attribution falls inside the
lesion segmentation, against the 26.5% a map without information would place there. Its peak lies
inside the lesion for 98% of images, against 26.5% by chance. It is the most concentrated of the
three methods, with a Gini index of 0.70 against 0.33 for random attribution.

**Model randomisation: passes.** Randomising only the classifier's output layer drops the mean rank
correlation with the original map to 0.11, and it stays low down to the stem. The highest single
value, 0.46, is seed 42 at that first step, below the threshold but not by much.

**Label randomisation: fails narrowly.** A fusion model retrained for 10 epochs on shuffled labels
memorised part of them (training macro-F1 0.37–0.40 against the shuffled labels) and scores at
chance on the true test labels (macro-F1 0.10–0.11,
[`label_randomisation.csv`](label_randomisation.csv)). Its Grad-CAM maps still correlate at 0.50
with those of the real model: 0.45, 0.49, and 0.57 across the three seeds. The last value crosses
the threshold. That is well below the 0.83 between two models trained on true labels, so the maps do
change. But a model that learned nothing about the diagnosis still produces maps correlating at 0.50
with the real ones. Part of what Grad-CAM shows is where the lesion is, the salient object any
network trained on these images attends to, rather than evidence for the class.

**Verdict: fails the label-randomisation check.** It is the best-behaved method here and a useful
coarse indicator of which region the model depends on. Its good localisation is partly
label-independent, and it says nothing about what in that region drove the class.

## Integrated gradients: responds to the image more than to the model

**Model randomisation: fails decisively.** After the classifier's output layer is randomised, the
integrated-gradients maps still correlate at 0.72 with the originals. The mean stays above 0.5
until the third image block from the top is randomised, and with the entire image branch random the
correlation is still 0.37. The maps are largely determined by the input: each pixel's attribution is
its difference from the baseline times an averaged gradient, and the first factor survives any
randomisation. This matches Adebayo et al.'s observation that methods which multiply by the input
keep the input's structure after the model is randomised.
[`figures/sanity_examples.png`](figures/sanity_examples.png) shows the same speckle pattern
persisting while Grad-CAM's map moves.

**Label randomisation: passes, but the pass is not reassuring.** The shuffled-label model's maps
correlate at 0.37 with the originals, under the threshold. Yet two models trained on the same true
labels agree at only 0.41. The maps differ about as much between two correct models as between a
correct model and one trained on noise. The change after label shuffling reflects instability, not
sensitivity to what the model learned.

**Faithfulness: fragile.** Deletion beats random with the blurred reference (−0.043) but not
consistently with the mean-colour reference (−0.020). Insertion is worse than random (−0.041).

**Localisation: close to chance.** 27% of the attribution falls inside the lesion, against the
lesion's 26.5% share of the image, so the map is spread over lesion and background skin alike. Its
peak lands in the lesion for 46% of images, above the 26.5% chance level but far below Grad-CAM's
98%. Its Gini index of 0.36 is barely above random attribution (0.33).

**Agreement with Grad-CAM: close to none.** The two image methods correlate at 0.12, and their
top-10% regions overlap by 0.13 intersection over union, against 0.05 for two random maps
([`method_agreement.csv`](method_agreement.csv)). They mostly highlight different parts of the same
image.

**Verdict: fails.** Integrated gradients, as configured here, is not a valid explanation of this
model's prediction.

## Attention rollout: explains the language model, not the classifier

**Model randomisation: fails.** Randomising both layers of the fusion classifier leaves the rollout
map unchanged (rank correlation 1.00), because rollout never reads the classifier. It stays above
0.5 until only the bottom DistilBERT layer and the embeddings keep their trained weights. Any
classifier on top of these text features, trained or random, gets the same explanation.

**Label randomisation: fails.** The shuffled-label model's rollout correlates at 0.89 with the real
model's, and two independently trained models at 0.97. Fine-tuning barely moves DistilBERT's
attention, so the map is fixed by the attention patterns, whatever the model learned to predict.

**Faithfulness: better than random, on a signal that barely exists.** Rollout beats random on
deletion (−0.015) and insertion (+0.012) in every seed. The margins are tiny because the text barely
moves the prediction at all: hiding every content token lowers the predicted-class probability only
from 0.92 to 0.85 on average. Rollout is also flatter than random noise (Gini 0.26 against 0.33). In
all eight cases in [`figures/explanations_correct.png`](figures/explanations_correct.png) and
[`figures/explanations_incorrect.png`](figures/explanations_incorrect.png), its largest weight is on
the closing full stop.

**Verdict: fails both sanity checks.** Rollout describes where DistilBERT's attention flows. It
carries no information about why the fusion model chose a class.

## What these checks did and did not show

- The deletion metric alone would have ranked all three methods as better than random, and a report
  built on it would have called all three faithful. The insertion metric and the two randomisation
  tests are what separated them.
- Localisation is not faithfulness. Grad-CAM's map sits on the lesion, and a model trained on
  shuffled labels keeps half of that map. A well-placed explanation is not evidence that the model
  used what is inside it for the reason it predicted.
- The verdicts hold for this model, this dataset, and these settings: 8-pixel patches, a blurred
  reference, a mean-colour baseline for integrated gradients, and a 0.5 similarity threshold. The
  threshold is a convention fixed in advance, not a derived quantity. Grad-CAM's label-randomisation
  result sits right at it and could flip with another seed.
- Deletion and insertion replace image regions with blurred content the model never saw in
  training. How much of Grad-CAM's poor insertion score is out-of-distribution behaviour rather than
  unfaithfulness cannot be separated with these tests. The mean-colour check shows the sign of the
  insertion result is stable, while its size depends on the reference.

The per-seed values are in [`faithfulness_runs.csv`](faithfulness_runs.csv) and
[`sanity_runs.csv`](sanity_runs.csv), the per-step randomisation curves in
[`sanity_checks.csv`](sanity_checks.csv), and the verdict logic's inputs in
[`sanity_verdicts.csv`](sanity_verdicts.csv).
