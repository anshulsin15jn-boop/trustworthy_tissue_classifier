# Summary: a tissue classifier that knows when to ask for help

## Pitch

I trained a convolutional network to classify nine types of colon tissue in H&E images, and it reaches 87.5% accuracy on slides from a hospital it never saw during training. I then tested whether it knows when it is wrong: letting it abstain on its most uncertain half raises accuracy to about 97%, but at the new hospital it is overconfident, and it confidently labels blood cell images as tissue background. The project shows what uncertainty methods such as Monte Carlo dropout can catch, and where a pathologist would still need a separate safety check.

## Question

A model that is accurate on average can still be confidently wrong on individual images, and in pathology the individual image is what matters. I asked whether a tissue classifier can flag the images it should not be trusted on, particularly when the slides come from a different hospital.

## Data

I used PathMNIST: 107,180 H&E stained colorectal tissue patches at 28 by 28 pixels in 9 classes. Training and validation images come from one source, and the 7,180 test images come from a different clinical center. I found that the test images differ in color because of staining or scanning, not just because of a different mix of tissue types.

## Results

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Logistic regression on flattened pixels | 0.574 | 0.448 |
| Random forest on flattened pixels | 0.640 | 0.562 |
| Small CNN, trained on a laptop CPU | **0.875** | **0.840** |
| ResNet18, pretrained, 4 CPU epochs (undertrained) | 0.839 | 0.793 |

1. **Spatial structure matters.** Models that treat pixels as an unordered list reach at most 0.640 accuracy; a small CNN reaches 0.875.
2. **A new hospital costs about 10 points.** The CNN scores 0.977 on validation (same source as training) and 0.875 on the external test set.
3. **Uncertainty is useful for triage.** Monte Carlo dropout entropy separates right from wrong predictions better than max softmax probability (AUROC 0.828 versus 0.774), and answering only the most certain half of images gives 0.969 accuracy.
4. **Confidence alone is not reliable.**
   * The model is well calibrated in house (ECE 0.003) but overconfident at the new hospital (ECE 0.064), and temperature scaling fitted in house does not help (0.069).
   * Under heavy noise, accuracy falls to chance while mean confidence stays between 0.79 and 0.89.
   * The model labels 98% of BloodMNIST images as background, 98.7% of them with over 90% confidence, and max softmax probability detects them worse than chance (AUROC 0.172).
5. **Explanations are partly sensible.** Grad CAM heatmaps pass two sanity checks and highlight plausible structures for several classes, but the most confident errors look like a shortcut based on overall color and texture.

## Limitations

Low resolution (28 by 28), a single public dataset with one external center, stained H&E rather than label free infrared imaging, simulated or cross domain out of distribution data, one training run per model, and an undertrained ResNet18.

## Next steps

Train ResNet18 properly on a GPU at higher resolution, add stain augmentation to narrow the hospital gap, evaluate ensembles for uncertainty, and add a dedicated check that rejects inputs that are not tissue.
