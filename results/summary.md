# Summary: a tissue classifier that knows when to ask for help

## The pitch (say it out loud)

I trained a convolutional network to classify nine types of colon tissue in H&E images, and it reaches 87.5% accuracy on images from a hospital it never saw. Then I tested whether it knows when it is wrong: letting it skip its most uncertain half raises accuracy to about 97%, but at the new hospital it is overconfident, and it confidently calls blood cell images "background". So the project shows both what uncertainty methods like Monte Carlo dropout can catch and where a pathologist still needs a separate safety check.

## The question

A model that is accurate on average can still be confidently wrong on individual images. In pathology that matters more than the average score. The question here is whether a tissue classifier can flag the images it should not be trusted on.

## The data

PathMNIST: 107,180 small (28 by 28 pixel) patches of H&E stained colorectal tissue in 9 classes. Training and validation images come from one source; the 7,180 test images come from a different clinical center. Their colors differ because of staining or scanning, not just because of a different mix of tissue types.

## What was built and what it found

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Logistic regression on raw pixels | 0.574 | 0.448 |
| Random forest on raw pixels | 0.640 | 0.562 |
| Small CNN, trained on a laptop CPU | **0.875** | **0.840** |
| ResNet18, pretrained, only 4 epochs on CPU (undertrained) | 0.839 | 0.793 |

1. **Structure matters.** Models that see pixels as an unordered list top out around 64% accuracy. A small CNN, which looks at local neighborhoods of pixels, reaches 87.5%.
2. **A new hospital costs about 10 points.** The CNN scores 97.7% on validation (same source as training) but 87.5% on test (new center).
3. **Uncertainty is useful.** Monte Carlo dropout (running the model 30 times with random parts switched off) separates right from wrong answers better than plain confidence (AUROC 0.83 versus 0.77). Answering only the most certain half gives 96.9% accuracy.
4. **Confidence alone is not trustworthy.**
   * The model is well calibrated in house (ECE 0.003) but about 6 points overconfident at the new hospital (ECE 0.064). Temperature scaling learned in house does not fix this.
   * Under heavy noise its accuracy falls to chance while it stays about 80 to 90% confident.
   * It labels 98% of blood cell images as "background", almost all with over 90% confidence. Plain confidence is worse than a coin flip at spotting them (AUROC 0.17).
5. **Explanations partly make sense.** Grad CAM heatmaps pass two sanity checks. For some classes they highlight sensible structures (cell walls in fat tissue, clusters of nuclei in lymphocytes), but the most confident mistakes look like a shortcut based on overall color and texture.

## Honest limitations

Very low resolution (28 by 28), one public dataset, H&E stained images rather than label free infrared imaging, simulated or cross domain unfamiliar data, one training run per model with no error bars, and an undertrained ResNet18.

## What I would do next

Train ResNet18 properly on a GPU at higher resolution, add stain augmentation to shrink the hospital gap, try ensembles for uncertainty, and add a dedicated "is this tissue at all?" check before the classifier.
