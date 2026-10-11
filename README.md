# Semi-Supervised Animal Classification under Domain Shift (ResNet-18)

Classifying 10 animal classes with only 779 labelled images, then improving the model with
14,800 unlabelled images using Noisy Student self-training. Built for a campus ML assessment,
with the restriction of using only ResNet-18.

## The problem
- 10 classes (Animals-10, Italian class names: cane = dog, gatto = cat, ragno = spider, etc.)
- 779 labelled images, 14,800 unlabelled images, 10,477 test images
- Phase 1: train on labelled data only
- Phase 2: use the unlabelled data as well

## Key finding: domain shift
Pseudo-labelling initially made the model worse. Investigating why showed that the
labelled images were **greyscale and upright**, while the unlabelled and test images
were **colour and rotated by 90°/180°/270°**.

Evidence:
- Model confidence was 0.74 on validation images but only 0.27 on unlabelled images
- The model labelled 6,033 of 14,800 unlabelled images as "cat"
- On rotated images, the Phase 1 model dropped from 90% to 65% accuracy

Fix:
- Convert all images to greyscale
- Train with random 90° rotations
- Average predictions over 4 orientations (× horizontal flip) at test time
- Select models on a rotated validation set that mirrors the test conditions

## Approach
**Phase 1:** ImageNet-pretrained ResNet-18, full fine-tuning, AdamW with one-cycle learning rate,
label smoothing, mixed precision, class-balanced sampling. Models selected by mean per-class accuracy.

**Phase 2 (Noisy Student):**
1. Train a rotation-robust teacher on the labelled images
2. Teacher pseudo-labels the 14,800 unlabelled images
3. Keep the most confident half of each predicted class (max 500) → 4,893 balanced pseudo-labels
4. Train a fresh student on labelled + pseudo-labelled images with RandAugment
5. Final prediction averages the student, teacher and Phase 1 model at 256×256 input

## Results
| Submission | Test accuracy |
|---|---|
| Phase 1 (labelled data only) | 80.46% |
| Phase 2 (labelled + unlabelled, 3 ResNet-18s averaged) | **86.31%** |

The student beat the teacher by about 4 points of mean per-class accuracy, and the rotated
validation set predicted test accuracy within 1–2 points.

## Experiment log
| Step | Change | Effect |
|---|---|---|
| Phase 1 baseline | Pretrained ResNet-18 | 88.5% val, but cat class only 60% |
| Class balancing | Balanced sampling, 30 epochs | Mean per-class 84.6% → 88.4% |
| Pseudo-labels, global threshold | Single confidence cut-off | 600 spiders, 0 squirrels; worse |
| Pseudo-labels, per-class selection | Top half per class | Still worse → led to finding the domain shift |
| Domain-shift fix | Greyscale + rotation training + rotated validation | Student +4 points over teacher |
| Ensemble + 256 px input | Average 3 models | Test 84.16% → 86.31% |

## Running it
Open the notebook in Google Colab with a T4 GPU. The dataset was provided for the assessment
and is not included in this repository.

## Tech stack
Python, PyTorch, torchvision, scikit-learn, pandas, Google Colab (T4 GPU)

## Next steps
- FixMatch-style consistency training
- Rotation prediction as a self-supervised pretraining task
