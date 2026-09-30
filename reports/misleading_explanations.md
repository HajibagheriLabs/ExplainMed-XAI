# When explanations mislead

A common argument for saliency maps in medical AI is that they make a model safe to use. A clinician
looks at the map, sees whether the model attended to the lesion, and catches the cases where it did
not. This document tests that argument on this repo's own model and finds that it does not hold. On
confident errors, the explanations look exactly as they do on correct predictions. That includes
melanomas called benign moles with near certainty.

Every number here comes from `scripts/misleading_explanations.py` (`make eval`). The script explains
every one of the 1,497 test images with each of the three fusion models (seeds 42, 43, 44). The
cases shown are from the seed-42 model, the same model used for the other explanation figures.

## How the cases were selected

The selection rule was written before looking at any case:

- **Confidently wrong:** the predicted class is wrong, and the model gives it a probability of at
  least 0.9.
- **Plausible explanation:** the Grad-CAM map's peak lies inside the lesion segmentation, and the
  share of the map inside the lesion is at least that of the *median correct prediction*. For the
  seed-42 model that median is 76%. By construction, the map looks at least as well placed as a
  typical map the model produces when it is right.

Grad-CAM is used for the rule because it is the method a reader would most readily trust. It
produces the cleanest, best-localised maps (69% of attribution inside the lesion, peak in the lesion
for 98% of images), and it is the only method that comes close to passing both sanity checks
([`explanation_analysis.md`](explanation_analysis.md)). Integrated gradients and attention rollout
already fail the sanity checks, so they could not be expected to flag anything.

The figure shows one case per true class, the most confident one, with missed malignancies
preferred and shown first. Every case meeting the rule is listed in
[`misleading_cases.csv`](misleading_cases.csv).

![Four confidently wrong predictions whose Grad-CAM maps sit on the lesion](figures/misleading_explanations.png)

## Four cases

**ISIC_0032936: a melanoma called a melanocytic nevus with probability 0.9997.** This is the most
dangerous error the model can make: it labels a malignant lesion as a common benign mole. Grad-CAM
places 80% of its attribution inside the lesion outline, well above the lesion's 50% share of the
image, with its peak inside. Nothing in the map distinguishes it from a correct nevus prediction.
The map for the predicted class sits on the lower right of the lesion, away from its darkest area on
the left. That detail is only visible because the lesion segmentation is drawn on top; a user shown
the map alone sees a heatmap on the lesion. The decisive observation is the third panel. Asked to
explain *melanoma* instead of *nevus*, Grad-CAM marks largely the same region: the two maps
correlate at 0.76. An explanation that looks the same whichever answer it is asked to justify cannot
tell a reader which answer is right.

**ISIC_0029820: a basal cell carcinoma called a benign keratosis with probability 0.995.** Grad-CAM
puts 93% of its attribution inside the lesion, as a clean central blob. By the standard of a
plausible explanation, this is a model looking at the right thing. Here the map for the true class
does differ (correlation 0.16). Grad-CAM can separate the classes on this image, but a reader shown
the explanation of the prediction, which is what an explanation interface shows, never sees that
difference.

**ISIC_0027829: an actinic keratosis or intraepithelial carcinoma called a benign keratosis with
probability 0.95.** 79% of the map lies inside a lesion that covers only 25% of the image. This
image also turned up in the seeded random sample of test predictions rendered in
[`figures/explanations_incorrect.png`](figures/explanations_incorrect.png), before this selection was
made. A well-localised map on a confident error is not a rare edge case one has to search for.

**ISIC_0031404: a melanocytic nevus called a basal cell carcinoma with probability 0.9999.** This is
the error in the other direction, a false alarm. 91% of the map is inside the lesion, and the map
for the true class correlates at 0.83 with it. The model would send a benign mole for an
unnecessary excision with near certainty and an explanation that looks well founded.

## How common this is

Per fusion model, over the 1,497 test images ([`misleading_counts.csv`](misleading_counts.csv)):

| Seed | Wrong | Confidently wrong | ...with a plausible explanation | ...of which a missed malignancy |
| ---: | ----: | ----------------: | ------------------------------: | ------------------------------: |
|   42 |   243 |                93 |                              44 |                              13 |
|   43 |   260 |                87 |                              41 |                              13 |
|   44 |   244 |                99 |                              47 |                              17 |

Close to half of all confident errors, 47% in every seed, come with a Grad-CAM map at least as well
placed as the median correct prediction's. For correct predictions the share is 50%, as the rule
implies. **The explanation is as convincing when the model is confidently wrong as when it is
right.**

The missed melanomas are the sharpest case. For the seed-42 model, 10 of the 44 misleading cases are
melanomas called nevi, each with a probability of at least 0.97 and between 78% and 97% of the map
inside the lesion. For every one of the ten, Grad-CAM's map for the true class, melanoma, correlates
with its map for nevus at between 0.76 and 0.96. On the very errors that matter most, the
explanation highlights the same place whichever class it is asked about.

## Can the explanation warn that the prediction is wrong?

If explanations make a model safer, a bad-looking explanation should at least signal a likely
error. Each signal below is scored by how well it separates wrong from correct test predictions, as
an AUROC. 0.5 means no information, and 1.0 means perfect separation
([`error_signals.csv`](error_signals.csv)):

| Signal                                                | AUROC for errors, mean ± sd over seeds |
| ----------------------------------------------------- | -------------------------------------: |
| Model confidence (probability of the predicted class) |                        0.8490 ± 0.0101 |
| Grad-CAM share of attribution inside the lesion       |                        0.5326 ± 0.0348 |
| Grad-CAM peak inside the lesion                       |                        0.5152 ± 0.0060 |
| Integrated gradients share inside the lesion          |                        0.5418 ± 0.0119 |

![Distributions of confidence and of attribution inside the lesion for correct and wrong predictions](figures/error_signals.png)

The explanation-based signals are close to chance. For one of the three models (seed 43), Grad-CAM's
localisation scores 0.49, slightly worse than no information at all
([`error_signals_runs.csv`](error_signals_runs.csv)). The model's own softmax probability, which
needs no explanation method at all, is a far better warning. It fails exactly where the danger lies:
confident errors are, by definition, the cases confidence does not flag. On those cases the
explanation is the only remaining check, and it looks no different from a correct prediction.

## What this means for "explanations make medical AI safe"

The argument assumes a saliency map tells the reader something about whether the prediction is
right. On this model, the evidence says it does not, for four reasons that come from the results
above rather than from general scepticism.

1. **A saliency map answers "where", not "why" or "whether".** For the missed melanomas, the maps
   for *nevus* and for *melanoma* cover nearly the same region (rank correlations 0.76 to 0.96). A
   map that points at the lesion is compatible with every diagnosis the lesion could have, so it
   cannot help a reader choose between them. Deciding whether a lesion is a melanoma is a question
   about what is in the region, and Grad-CAM only says which region was used.
2. **Good localisation is not evidence of correctness.** Across 1,497 test images and three models,
   the share of Grad-CAM attribution inside the lesion separates wrong from correct predictions at
   an AUROC of 0.53. A reviewer who accepts predictions whose map is on the lesion accepts confident
   errors about as often as correct ones.
3. **Part of the plausibility is not about the model's reasoning at all.** A fusion model retrained
   on shuffled labels, which predicts at chance on the true labels, still produces Grad-CAM maps that
   correlate at 0.50 with the real model's ([`explanation_analysis.md`](explanation_analysis.md)).
   Maps land on the lesion largely because the lesion is the salient object in a dermatoscopic image,
   and any network trained on these images attends to it. A heatmap on the lesion is the expected
   output of a model that learned something, and also of one that learned nothing.
4. **The more detailed-looking methods are worse, not better.** Integrated gradients produces fine,
   pixel-level maps that look like careful analysis, yet it fails the model-randomisation check: its
   maps survive the destruction of the classifier. The text explanation, attention rollout, is
   unchanged by randomising the classifier at all. A clinician offered three explanation views
   would receive one weakly informative map and two that do not reflect the decision.

The practical risk runs opposite to the promise. A confident prediction with a well-placed heatmap is
more persuasive than a bare prediction. The 41 to 47 confidently wrong, plausibly explained cases per
model are exactly those in which a human reviewer would be most likely to defer. For the seed-42
model, 13 of them are malignant lesions called benign. That is an argument about how people use such
displays, not a measured result: this repo includes no reader study. The measurements show that
nothing in the map would stop the deference.

What the results do support is narrower. Calibrated confidence carries real information about
errors (AUROC 0.85), and a per-class error rate is honest about the risk: melanoma recall is 0.69.
Sanity-checked explanation methods can still help debug a model at the level of a dataset, for
example to find a model attending to rulers or ink marks. None of that is the per-patient
reassurance the argument promises.

## Limitations of this analysis

- "Plausible" is operationalised with the lesion segmentation and a fixed rule, not rated by
  dermatologists. A clinician might notice things in these maps that the rule does not capture, or
  might trust them for reasons it does not model.
- One dataset, one architecture, and three training seeds. The rates would differ for another model.
  The mechanism, maps that depend more on where the lesion is than on what the model concluded, is
  the part expected to generalise, and the sanity checks measure it directly.
- The segmentation masks are themselves annotations and can be imprecise at the lesion border.
