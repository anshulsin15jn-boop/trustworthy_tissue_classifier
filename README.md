# Trustworthy Tissue Image Classifier

A tissue image classifier that flags when it is unsure or seeing unfamiliar data, so a pathologist would know when to double check it.

## The question

Medical image classifiers can be accurate on average and still be confidently wrong on individual images, especially images that look different from their training data. This project trains classifiers on colon tissue images and then asks how far the model can be trusted: does it know when it does not know?

## Results at a glance

* A small CNN reaches **0.875 test accuracy and 0.840 macro F1**, far above flattened pixel baselines (best 0.640 and 0.562).
* It scores 0.977 on validation but 0.875 on test. The test images come from another hospital, and Phase 3 shows their colors differ because of staining, not because of a different class mix.
* **Uncertainty helps:** if the model answers only its most certain half of test images, accuracy rises to 96.9% using Monte Carlo dropout.
* **Confidence is not trustworthy on its own:** the model is about 6 points overconfident at the new hospital, temperature scaling fitted in house does not fix it, and it labels blood cell images as "background" with over 90% confidence 98.7% of the time.

One page summary: [`results/summary.md`](results/summary.md).

## The data

[PathMNIST](https://medmnist.com/) from the MedMNIST collection: 107,180 patches of H&E stained colorectal cancer tissue, each 28 by 28 pixels, in 9 tissue classes. The official splits are 89,996 training, 10,004 validation, and 7,180 test images. The test images come from a different clinical center than the training images. See `notebooks/00_explore_data.ipynb` for the class balance, example images, and pixel statistics.

![Example images per class](results/figures/examples_per_class.png)

### What we learned from the data

* **The training classes are fairly balanced.** The most common class (adenocarcinoma epithelium, 14.3%) is only about 1.6 times as common as the least common (normal colon mucosa, 8.8%). We still report macro F1 and per class recall, which weigh every class equally.
* **The "background" class is not just empty glass.** Many background patches are dark, blurry, or edge regions of the slide. Some could be mistaken for dense purple tissue such as lymphocytes.
* **The test set differs from the training set in two ways at once.** It comes from a different clinical center, and its pixel colors are more spread out. It also has a different mix of classes: 18.6% of test images are adipose (fat) versus 10.4% in training. Adipose patches are mostly pale, so the class mix alone shifts the color statistics. These overall histograms cannot tell us how much of the difference is staining and how much is class mix; Phase 3 answers that with a fairer comparison (it is staining).
* **Validation looks like training, not like test.** Validation was cut from the same source as training, so a model may score higher on validation than on test. That gap is something to watch in later phases.

## Methods by phase

| Phase | What | Status |
|---|---|---|
| 0 | Setup and data exploration | done |
| 1 | Baselines: logistic regression, random forest | done |
| 2 | Small CNN and pretrained ResNet18, both trained on CPU | done |
| 3 | Trust: calibration, uncertainty, out of distribution detection, staining check | done |
| 4 | Explainability with Grad CAM, writeup | done |

## Results

Filled in as each phase finishes. Every number here comes from code in this repo.

### Phase 1: baselines on flattened pixels

Both models see each image as 2,352 unordered numbers and were trained on all 89,996 training images. Logistic regression's regularization strength was chosen on the validation set. Run `python -m src.baselines`; the log is in `results/logs/baselines.txt`.

| Model | Test accuracy | Test macro F1 | Validation accuracy |
|---|---|---|---|
| Logistic regression (C = 0.0001) | 0.574 | 0.448 | 0.527 |
| Random forest (200 trees) | 0.640 | 0.562 | 0.672 |

What we learned:

* **Flattened pixels only get you so far.** Background (100% recall for the random forest) and adipose are easy because their overall color is distinctive. Classes defined by texture and structure are hard.
* **Normal mucosa is the worst case.** The random forest finds only 25% of normal mucosa and calls 48% of it tumor; logistic regression finds only 7%. A model that calls healthy tissue cancer is exactly the kind of error a pathologist needs flagged.
* **Accuracy and macro F1 can move in opposite directions.** Logistic regression scores higher accuracy on test than on validation (0.574 versus 0.527) but lower macro F1 (0.448 versus 0.480). The test set contains more of the easy adipose class, which lifts accuracy without the model being better at the hard classes.
* **Why a CNN should help:** flattening throws away which pixels are next to each other, so a model cannot learn "a ring of purple nuclei around a pale gland." A convolutional network keeps the 2D layout and learns small local patterns.

![Random forest confusion matrix](results/figures/random_forest_confusion.png)

### Phase 2: convolutional neural networks

All four models on the same official test set (`python -m src.compare` rebuilds this table from `results/metrics/`):

| Model | Test accuracy | Test macro F1 | Validation accuracy |
|---|---|---|---|
| Logistic regression (flattened pixels) | 0.574 | 0.448 | 0.527 |
| Random forest (flattened pixels) | 0.640 | 0.562 | 0.672 |
| **Small CNN** (trained from scratch, CPU) | **0.875** | **0.840** | 0.977 |
| ResNet18 (ImageNet pretrained, fine tuned on CPU) | 0.839 | 0.793 | 0.925 |

* **Small CNN:** three convolution blocks and a dropout classifier, 111,369 parameters. Trained with random flips and 90 degree rotations, Adam, batch size 128, and early stopping on validation loss (patience 4). It stopped after 28 epochs and kept epoch 24. Training took 22.1 minutes on a laptop CPU with a peak of 1.14 GB of RAM. Log: `results/logs/small_cnn.txt`.
* **ResNet18:** ImageNet weights with a new 9 class output layer, fine tuned at 28 by 28. Each epoch took about 6 to 8 minutes on the CPU, so it ran under a 25 minute budget: 4 epochs, keeping epoch 3. Peak RAM was 1.04 GB. Log: `results/logs/resnet18.txt`.

What we learned:

* **Keeping the 2D layout is what matters.** The small CNN beats the random forest by 23.5 points of accuracy and 0.28 of macro F1. Normal mucosa recall rose from 25% to 91%.
* **ResNet18 is undertrained, so its lower score says nothing about pretrained models in general.** It trained for only 4 epochs on a laptop CPU (the small CNN had 28), and it was still improving. Its design also shrinks a 28 by 28 input to a single cell by the last block, far from the 224 by 224 photos it was pretrained on. A longer run on a GPU, ideally at 64 by 64 or 224 by 224, is a planned next step; until then the fair conclusion is "not tested properly yet", not "pretraining does not help".
* **There is a big gap between validation and test.** The small CNN reaches 0.977 on validation but 0.875 on test. Validation comes from the same source as training; test comes from another clinical center. A model can look nearly perfect in house and still drop 10 points at a new hospital.
* **The hardest classes on test** are cancer associated stroma (55% recall, often called debris or smooth muscle), smooth muscle (67%), and mucus (74%, sometimes called background or adipose).

![Small CNN training curves](results/figures/small_cnn_training_curves.png)

![Small CNN confusion matrix](results/figures/small_cnn_confusion.png)

## Trust

Phase 3 asks whether the small CNN knows when it does not know. Run `python -m src.trust` and `python -m src.stain`; all numbers are in `results/metrics/trust.json` and `results/metrics/stain.json`.

### Calibration: does 90 percent confidence mean 90 percent right?

| | Validation ECE | Test ECE | Mean confidence on test | Test accuracy |
|---|---|---|---|---|
| Small CNN | 0.003 | 0.064 | 0.938 | 0.875 |
| Small CNN after temperature scaling (T = 0.93) | 0.003 | 0.069 | 0.942 | 0.875 |

* **In house, the model is almost perfectly calibrated** (ECE 0.003 on validation). **At the new hospital it is overconfident**: about 6 points more confident than it is accurate.
* **Temperature scaling did not help.** It is fitted on validation data, where there was nothing to fix, so it learned T = 0.93 (slightly sharper) and made test calibration a little worse. A calibration fix learned on in house data cannot anticipate a shift it has never seen. (For ResNet18, using the local checkpoint, temperature scaling did lower test ECE slightly, from 0.078 to 0.071.)

![Reliability diagram](results/figures/reliability_diagram.png)

### Uncertainty: can the model hand its hardest cases to a pathologist?

| Uncertainty score | Accuracy if the model answers its most certain 50% | 80% | 90% | All |
|---|---|---|---|---|
| Max softmax probability | 0.950 | 0.935 | 0.916 | 0.875 |
| Monte Carlo dropout entropy (30 passes) | **0.969** | **0.940** | 0.910 | 0.873 |

* **Refusing uncertain images works.** If the model answers only its most certain half, accuracy rises from 87.5% to 95.0% with max softmax and 96.9% with Monte Carlo dropout.
* **Monte Carlo dropout separates right from wrong answers best**: AUROC 0.828, against 0.774 for max softmax probability. Its advantage is at low and middle coverage; at 90% coverage the two are about equal.
* **Some mistakes are invisible to confidence:** 219 wrong test predictions had more than 99% confidence. The most confident ones are all smooth muscle called background at 100%.

![Accuracy versus coverage](results/figures/accuracy_coverage.png)

![Confident mistakes](results/figures/confident_mistakes.png)

### Out of distribution detection: does the model notice unfamiliar images?

AUROC for telling clean test images apart from each kind of unfamiliar data (0.5 is chance, 1.0 is perfect):

| Unfamiliar data | Accuracy on it | Max softmax | Monte Carlo dropout entropy |
|---|---|---|---|
| Gaussian noise (strongest) | 0.092 | 0.627 | **0.957** |
| Brightness (strongest) | 0.089 | 0.714 | **0.871** |
| Blur (strongest) | 0.295 | 0.580 | 0.629 |
| Contrast (strongest) | 0.336 | 0.548 | 0.597 |
| BloodMNIST (blood cells, not tissue) | not applicable | **0.172** | 0.536 |

* **Corruptions wreck accuracy while confidence stays high.** Under noise, accuracy falls to about 9%, which is chance for 9 classes, yet mean confidence stays between 0.79 and 0.89.
* **Monte Carlo dropout catches noise and brightness changes well** (AUROC up to 0.96). Max softmax catches them only weakly, and neither catches faded contrast or blur.
* **BloodMNIST is a complete failure.** The tissue model calls 98% of blood cell images "background", and 98.7% of them get over 90% confidence. Max softmax AUROC is 0.17, worse than chance: the model is *more* confident on blood cells than on real tissue. Monte Carlo dropout only brings this to 0.54, close to chance. A softmax classifier has no "none of the above" option, so data from a different domain can land confidently in whichever class it most resembles.

![Out of distribution AUROC](results/figures/ood_auroc.png)

![BloodMNIST images the model was confident about](results/figures/bloodmnist_confident.png)

### Is the test set's color different because of staining or because of class mix?

Phase 0 could not separate the two. Here the training set is reweighted to the test set's exact class mix. The color distance between train and test barely moves (red 0.173 to 0.171, green 0.186 to 0.196, blue 0.192 to 0.184, on a scale where 0 is identical and 1 is no overlap). **The difference is in the images themselves, not the class mix:** a real staining or scanning shift. Within single classes it is largest for background (0.62) and adipose (0.45) and smallest for normal mucosa (0.07).

![Color after matching class mix](results/figures/stain_reweighted_histograms.png)

### What worked and what did not

* **Worked:** Monte Carlo dropout entropy is the most useful flag. It raises accuracy on the cases the model keeps and detects noise and exposure problems well.
* **Did not work:** temperature scaling fitted in house did not fix overconfidence at a new hospital, contrast and blur changes were mostly undetected, and blood cell images, a different domain entirely, were not flagged at all.
* **The takeaway for a pathologist:** high confidence from this model is not a guarantee. Uncertainty flags catch some problems, but a separate check that an image is even tissue is needed before trusting any prediction.

## Explainability: where does the model look?

Grad CAM highlights the regions that pushed the small CNN toward its prediction (`python -m src.explain`; numbers in `results/metrics/gradcam.json`).

![Grad CAM examples](results/figures/gradcam_examples.png)

* **Sanity checks pass.** Heatmaps from the trained model are unrelated to heatmaps from an untrained copy (median correlation −0.02 over 500 test images), so they reflect what the model learned rather than just image edges. Heatmaps for the predicted class and the second most likely class also differ (median correlation −0.13), so they are specific to the class being explained.
* **Several correct predictions look sensible:** for adipose it lights up the thin cell walls rather than the empty fat, for lymphocytes the dense clusters of nuclei, and for normal mucosa the ring of tissue around a gland.
* **The worst mistakes look like a texture shortcut.** When muscle or mucus is called background at 99 to 100% confidence, the heatmap covers almost the whole patch evenly instead of any one structure, which suggests the model is reacting to a smooth overall color and texture.
* **Uncertain images are mostly pale and sparse,** and their heatmaps are faint and scattered at the edges: the model finds little evidence for any class.
* **Caution:** the heatmap is computed on a 7 by 7 grid and stretched to 28 by 28, so it is coarse. It shows *where* the evidence was, not *why*, and it cannot prove the model uses the same features a pathologist would.

## Limitations

* **Low resolution:** 28 by 28 pixel patches lose most cell level detail that a pathologist would use.
* **One dataset from one source:** training data comes from one collection, and the test set from one other clinical center. Results at other hospitals could differ.
* **Stained images, not the lab's modality:** these are H&E stained images, not the label free infrared chemical images used by the Bhargava lab, so none of these numbers transfer directly.
* **Out of distribution data is simulated or from a different domain:** corruptions were added by code, and BloodMNIST is a completely different kind of image. Real failures (folded tissue, pen marks, a new scanner) were not tested.
* **ResNet18 is undertrained:** 4 epochs on a CPU, so its comparison with the small CNN is not fair yet.
* **One training run per model:** no repeated seeds, so there are no error bars, and small differences (for example 0.950 versus 0.969) may partly be noise.
* **Validation was used for several choices** (early stopping, the logistic regression C, and the temperature), and it does not resemble the test hospital.
* **Small CNN training took 22 minutes,** above the 15 minute target for laptop runs.

## What to try next

* Train ResNet18 properly on a GPU (Google Colab) for more epochs at 64 by 64 or 224 by 224.
* Reduce the stain shift: stain normalization or color augmentation during training, then check whether the validation to test gap shrinks.
* Better uncertainty: deep ensembles (several independently trained models), and dropout in the convolution layers, not only the classifier head.
* A dedicated "is this even tissue?" check, for example distance in the model's feature space, since softmax confidence failed on blood cells.
* Repeat every run with several seeds and report the spread.

## How to reproduce

Requires Python 3.12 and git. Name the Python version explicitly when creating the environment, because plain `python` may point to a different version on your machine.

```bash
git clone https://github.com/anshulsin15jn-boop/trustworthy_tissue_classifier.git
cd trustworthy_tissue_classifier

# Windows (PowerShell):
py -3.12 -m venv .venv
.venv\Scripts\activate
# macOS/Linux:
python3.12 -m venv .venv
source .venv/bin/activate

python --version            # should print Python 3.12.x
pip install -r requirements.txt
python -m pytest            # downloads PathMNIST (about 200 MB) on the first run
jupyter notebook notebooks/00_explore_data.ipynb
```

Then run each phase from the project root. Times are from a 16 core laptop CPU.

```bash
python -m src.baselines                      # Phase 1, about 3 minutes
python -m src.train --model small_cnn        # Phase 2, about 22 minutes (overwrites the committed checkpoint)
python -m src.train --model resnet18 --lr 0.0005 --max-epochs 5 --patience 2 --max-minutes 25   # about 27 minutes
python -m src.compare                        # comparison table
python -m src.stain                          # Phase 3 staining check, about 1 minute
python -m src.trust                          # Phase 3 trust analysis, a few minutes (downloads BloodMNIST)
python -m src.explain                        # Phase 4 Grad CAM, under a minute
```

Phases 3 and 4 only need the committed small CNN checkpoint, so you can skip the training commands and get the same numbers as this README. Retraining on another machine may give slightly different numbers, because CPU arithmetic is not perfectly deterministic. Run logs are in `results/logs/`.

## Engineering notes

* **Memory:** the first small CNN run was stopped when the laptop ran low on memory. Training now keeps images as 8 bit integers and normalizes one batch at a time, computes channel statistics in chunks, and uses a single process DataLoader (`num_workers=0`). Peak RAM fell to 1.14 GB for the small CNN and 1.04 GB for ResNet18, logged in each run.
* **Logistic regression** is fit with full batch `L-BFGS` in PyTorch, minimizing the same objective as scikit learn's `LogisticRegression`. scikit learn's own solver did not converge in reasonable time on 2,352 features; `tests/test_baselines.py` checks that the two agree.
* **Checkpoints:** the small CNN checkpoint (0.46 MB) is committed so Phases 3 and 4 use exactly the model reported here. The ResNet18 checkpoint (45 MB) is not committed.

## Project layout

```
src/
  data.py       loading PathMNIST and BloodMNIST, tensors, augmentation, corruptions
  baselines.py  Phase 1: logistic regression and random forest
  models.py     small CNN and ResNet18, checkpoints
  train.py      training loop with early stopping, test evaluation
  evaluate.py   accuracy, macro F1, per class metrics, confusion matrices
  compare.py    model comparison table
  trust.py      Phase 3: calibration, uncertainty, out of distribution detection
  stain.py      Phase 3: staining versus class mix comparison
  explain.py    Phase 4: Grad CAM and its sanity checks
  plotting.py   shared plot style
notebooks/      data exploration notebook
results/        metrics (JSON, CSV), figures, run logs, summary.md
checkpoints/    small_cnn.pt (committed); other checkpoints are not committed
tests/          pytest tests (25)
data/           downloaded datasets (not committed)
LEARNING_GUIDE.md  plain language guide to every part of the project
```
