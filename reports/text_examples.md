# The text modality

HAM10000 has no clinical notes. The text branch of every model in this repo reads a sentence
built by a fixed template from the structured metadata of each image. It is synthetic, it is not
clinical documentation, and results on it say nothing about real clinical text.

This file is written by `scripts/text_examples.py`; the template itself is
`src/explainmed/text.py`.

## The template

```
<patient> with a skin lesion <site>, imaged by dermatoscopy.[ The diagnosis was <method>.]
```

`<patient>` is `<age>-year-old <sex>` when both are recorded, and says "unrecorded age" or
"unrecorded sex" otherwise. The diagnosis itself is never part of the text.

| `localization` value | Phrase |
| --- | --- |
| abdomen | on the abdomen |
| acral | on an acral site |
| back | on the back |
| chest | on the chest |
| ear | on the ear |
| face | on the face |
| foot | on the foot |
| genital | in the genital area |
| hand | on the hand |
| lower extremity | on the lower extremity |
| neck | on the neck |
| scalp | on the scalp |
| trunk | on the trunk |
| upper extremity | on the upper extremity |
| unknown | at an unrecorded body site |

The bracketed diagnosis-method sentence is only present in one ablation (see below):

| `dx_type` value | Sentence |
| --- | --- |
| histo | The diagnosis was confirmed by histopathology. |
| follow_up | The diagnosis was established by follow-up examination. |
| consensus | The diagnosis was established by expert consensus. |
| confocal | The diagnosis was confirmed by in-vivo confocal microscopy. |

## Examples

One image per class, sampled with seed 42, plus the first images with a missing age, sex, or
body site.

| Image | Diagnosis (the label, never in the text) | Text the model reads | Sentence added in the diagnosis-method variant |
| --- | --- | --- | --- |
| ISIC_0030375 | actinic keratosis / intraepithelial carcinoma (`akiec`) | 65-year-old male with a skin lesion on the face, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0029951 | basal cell carcinoma (`bcc`) | 75-year-old male with a skin lesion on the lower extremity, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0029217 | benign keratosis-like lesion (`bkl`) | 35-year-old male with a skin lesion on the face, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0031309 | dermatofibroma (`df`) | 55-year-old male with a skin lesion on the lower extremity, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0033299 | melanoma (`mel`) | 85-year-old male with a skin lesion on the trunk, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0026036 | melanocytic nevus (`nv`) | 40-year-old male with a skin lesion on the abdomen, imaged by dermatoscopy. | The diagnosis was established by follow-up examination. |
| ISIC_0033565 | vascular lesion (`vasc`) | 25-year-old female with a skin lesion on the back, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0025009 | melanocytic nevus (`nv`) | Female patient of unrecorded age with a skin lesion on the chest, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0032529 | melanocytic nevus (`nv`) | 60-year-old patient of unrecorded sex with a skin lesion on the foot, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |
| ISIC_0024332 | basal cell carcinoma (`bcc`) | 60-year-old male with a skin lesion at an unrecorded body site, imaged by dermatoscopy. | The diagnosis was confirmed by histopathology. |

## Why the diagnosis method is left out

The diagnosis method records how the ground truth was established, which depends on the diagnosis
itself: suspicious lesions are excised and go to histopathology, benign-looking ones are followed
up. It would not be known when a prediction is made, and it nearly determines the label:

- every image whose diagnosis was confirmed by in-vivo confocal microscopy is `bkl` (69 images)
- every image whose diagnosis was established by follow-up examination is `nv` (3,704 images)
- every image of `akiec`, `bcc`, `mel` was confirmed by histopathology

Images per diagnosis method and diagnosis:

| `dx_type` | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confocal | 0 | 0 | 69 | 0 | 0 | 0 | 0 |
| consensus | 0 | 0 | 264 | 60 | 0 | 503 | 75 |
| follow_up | 0 | 0 | 0 | 0 | 0 | 3704 | 0 |
| histo | 327 | 514 | 766 | 55 | 1113 | 2498 | 67 |

The main text modality therefore excludes it. The text-only baseline is trained both with and
without it, so the size of the leak is measured instead of hidden.

## How much the text can carry

The 10,015 images map to only 419 distinct descriptions without the
diagnosis method (806 with it). 91% of images have a description that also
belongs to an image of a different diagnosis, which caps what any text-only model can achieve.

A lookup table that memorises how often each exact training description occurs in each class, and
predicts the class where it is relatively most frequent, reaches a test macro-F1 of
0.2093 and a balanced accuracy of 0.3054
(0.2906 and 0.3457 with the
diagnosis method). This is a reference for how much a text-only model can extract, not a
baseline to beat: descriptions unseen in training fall back to the most common class.
