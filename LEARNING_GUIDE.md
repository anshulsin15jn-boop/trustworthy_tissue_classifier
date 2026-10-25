# Learning Guide: Trustworthy Tissue Image Classifier

This guide explains every part of the project in plain language. It assumes you know basic Python and intro statistics, but not deep learning. Every number here comes from a saved file in `results/`.

## 1. The story in one paragraph

Pathologists diagnose cancer by looking at stained tissue under a microscope, and machine learning models can help sort tissue types automatically. But a model that is right 90% of the time on average can still be completely, confidently wrong on a particular image, and in medicine that single image might be the one that matters. This project trains models to classify nine types of colon tissue, then asks the harder question: **does the model know when it does not know?** It measures whether the model's confidence can be believed, whether it can set aside the images it is unsure about, and whether it notices images unlike anything it was trained on. The answer is mixed, and the honest mixed answer is the point: uncertainty methods help in some situations and fail badly in others, so a pathologist would still need a separate safety check.

## 2. Walkthrough of each phase

### Phase 0: setup and getting to know the data

**What we did.** Set up a Python 3.12 virtual environment with pinned package versions, created the GitHub repo, and wrote `src/data.py` so every later step loads the data the same way. The exploration notebook counts images per class, shows examples, and looks at pixel values.

**Why.** You cannot judge a model without knowing the data. If one class were 90% of the images, a model could score 90% by always guessing it.

**What we found.**
* 89,996 training, 10,004 validation, and 7,180 test images, each 28 by 28 pixels with 3 color channels (values 0 to 255).
* Training classes are fairly balanced: the most common class (tumor epithelium, 14.3%) is only 1.63 times as common as the least common (normal mucosa, 8.8%).
* **The test set comes from a different hospital** and looks different. It has a different class mix (18.6% adipose versus 10.4% in training; 4.7% debris versus 11.5%) and more spread out colors.
* The "background" class is not just empty glass: many background patches are dark and blurry.

**Phase 0 discussion points** (these were the Phase 0 quiz questions; good to be able to answer):
1. *Why use a virtual environment and pinned versions?* So a lab member who clones the repo gets exactly the same packages. Without pins, a newer library version might change a default setting or remove a function, so the code crashes or silently gives different numbers.
2. *If a model gets 99% on validation and 88% on test, what could explain it?* Two causes we found in Phase 0: the test images come from a different hospital (a staining or scanner shift), and the test set has a different class mix. This turned out to be real: the small CNN scored 97.7% on validation and 87.5% on test.
3. *How could more adipose in the test set make test accuracy look better than the model really is?* Adipose is easy (pale, distinctive). If the test set has more easy images, overall accuracy rises even if the model is no better at hard, clinically important classes like tumor or stroma. Macro F1 protects against this because it weighs every class equally. We actually saw this: logistic regression had higher accuracy on test than validation (0.574 versus 0.527) but lower macro F1 (0.448 versus 0.480).
4. *How could you separate staining from class mix?* Compare colors within one class at a time (adipose versus adipose), or reweight the training set so its class mix matches the test set and compare again. Phase 3 did both.

### Phase 1: simple baselines

**What we did.** Flattened each image into one row of 2,352 numbers (28 × 28 × 3) and trained logistic regression and a random forest on all 89,996 training images.

**Why.** Baselines tell you what "good" means. If deep learning only matched a random forest, it would not be worth its complexity.

**What we found.**

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Logistic regression | 0.574 | 0.448 |
| Random forest | 0.640 | 0.562 |

Easy classes (background, adipose) are recognized by overall color. Classes defined by structure are hard: the random forest finds only 25% of normal mucosa and calls 48% of it tumor. Flattening throws away which pixels are neighbors, so these models cannot see shapes.

### Phase 2: convolutional neural networks

**What we did.** Built a small CNN (111,369 adjustable numbers called parameters) and trained it on the laptop CPU with early stopping. Also fine tuned a pretrained ResNet18 (11.2 million parameters).

**Why.** CNNs keep the 2D layout of an image and learn local patterns, which is exactly what flattening destroyed.

**What we found.**

| Model | Test accuracy | Test macro F1 | Validation accuracy |
|---|---|---|---|
| Small CNN | **0.875** | **0.840** | 0.977 |
| ResNet18 (4 epochs only) | 0.839 | 0.793 | 0.925 |

* The small CNN beats the random forest by 23.5 points. Normal mucosa recall jumped from 25% to 91%.
* **The 10 point drop from validation (97.7%) to test (87.5%) is one of the most important results.** In house the model looks nearly perfect; at a new hospital it loses 10 points. A lab that only checked its own validation data would badly overestimate how the model performs elsewhere.
* Hardest test classes: cancer associated stroma (55% recall), smooth muscle (67%), mucus (74%).
* **ResNet18 is undertrained.** It ran only 4 epochs on a CPU (about 6 to 8 minutes each) and was still improving. Its lower score does **not** show that pretrained models are worse here; it shows we did not train it long enough. A longer run on a GPU in Google Colab is a next step.

### Phase 3: can the model be trusted?

**Calibration.** In house, the small CNN is almost perfectly calibrated (ECE 0.003 on validation). At the new hospital it is overconfident: average confidence 93.8% but accuracy 87.5% (ECE 0.064). Temperature scaling, fitted on validation, learned T = 0.93 and made test calibration slightly worse (ECE 0.069), because there was nothing to fix on validation.

**Uncertainty.** Monte Carlo dropout separates right answers from wrong ones better than plain confidence (AUROC 0.828 versus 0.774). If the model answers only its most certain 50% of test images, accuracy is 96.9% with Monte Carlo dropout and 95.0% with plain confidence. But 219 wrong predictions had over 99% confidence; the most confident ones were all smooth muscle called background.

**Out of distribution detection.**
* Corruptions destroy accuracy while confidence stays high: with heavy noise, accuracy is about 9% (chance for 9 classes is 11%), yet mean confidence is 79 to 89%.
* Monte Carlo dropout detects noise (AUROC up to 0.957) and strong brightness changes (0.871) well. Nothing detected faded contrast or blur well (AUROC 0.50 to 0.65).
* **BloodMNIST, a completely different kind of image, was not detected at all.** The model called 3,361 of 3,421 blood images "background", and 98.7% of them got over 90% confidence. Plain confidence scored AUROC 0.172, worse than a coin flip, because the model was *more* confident on blood cells than on real tissue. Monte Carlo dropout only reached 0.536.

**Staining versus class mix.** Reweighting the training images to the test set's class mix barely changed the color difference (distance 0.17 to 0.20 in every channel, before and after). So the test hospital's images really are colored differently: a staining or scanning shift.

### Phase 4: explainability and writeup

Grad CAM heatmaps show where the small CNN found evidence. Two sanity checks pass: heatmaps from the trained model are unrelated to those from an untrained copy (median correlation −0.02), and heatmaps differ between the predicted class and the runner up (−0.13). For adipose the model looks at cell walls, for lymphocytes at clusters of nuclei, and for normal mucosa at tissue around glands, all sensible. The worst mistakes (muscle called background at 100%) light up the whole patch evenly, suggesting a shortcut based on overall color and texture.

## 3. Key concepts, explained simply

**Train, validation, and test splits.** Like studying for an exam. The training set is the textbook the model learns from. The validation set is a practice exam used to make decisions (when to stop training, which settings to use). The test set is the real exam, opened once at the end. If you tune on the test set, you are peeking at the exam, and the score stops meaning anything. *Example:* we picked the logistic regression setting C = 0.0001 because it scored best on validation, never by looking at test.

**Stain variation.** H&E staining uses two dyes: hematoxylin turns nuclei purple, eosin turns cytoplasm pink. Each hospital's chemicals, timing, and scanner differ slightly, so the same tissue type can look a different shade. *Example:* our test hospital's muscle looks salmon red, and the model confidently called those patches background.

**Flattening.** Turning a 28 by 28 by 3 image into one list of 2,352 numbers. The model then has no idea which pixels are neighbors. *Example:* a ring of nuclei around a gland and the same pixels scattered randomly would look equally meaningful to it.

**Convolutional neural network (CNN).** A model that slides small filters (here 3 by 3 pixels) across the image. Each filter learns to detect a pattern, such as an edge or a purple dot. Later layers combine these into bigger patterns, such as a cluster of nuclei. Pooling layers shrink the image (28 → 14 → 7) so later layers see larger areas.

**Logits and softmax.** The model outputs one raw score (logit) per class. Softmax turns them into probabilities that add to 1. *Example:* logits (2, 1, 0) become about (0.67, 0.24, 0.09). The highest probability is the "confidence".

**Loss and gradient descent.** The loss measures how wrong the model is; we used cross entropy, which is small when the model gives high probability to the right class. Gradient descent adjusts each parameter slightly in the direction that lowers the loss, like walking downhill in fog by feeling the slope under your feet. An **epoch** is one full pass through the training set; a **mini batch** is the 128 images used for one step.

**Overfitting and early stopping.** Overfitting is memorizing the training images instead of learning general patterns, like memorizing answers to practice questions. Early stopping watches the validation loss after each epoch, saves the best model, and stops after 4 epochs without improvement. *Example:* the small CNN stopped after epoch 28 and kept epoch 24.

**Dropout.** During training, randomly switch off some neurons (here 30%) on each step, so the model cannot rely on any single one. It is like a team practicing with random players sitting out, so everyone learns to contribute.

**Data augmentation.** Showing flipped and rotated copies of training images. A tissue patch has no natural "up", so a rotated patch is still the same class. It gives more varied training data for free.

**Transfer learning.** Starting from a model already trained on a huge dataset (ResNet18 learned from 1.2 million everyday photos) and adjusting it to a new task. Early layers that detect edges and textures are useful almost everywhere. *Caveat from this project:* it still needs enough training, and our 28 by 28 images are far smaller than the photos it learned from.

**Accuracy versus macro F1.** Accuracy is the share of all images classified correctly, so common classes dominate it. **Precision** for a class: of the images the model called "tumor", how many were tumor? **Recall**: of the real tumor images, how many did the model find? **F1** combines the two. **Macro F1** averages F1 over the 9 classes, so every class counts equally. *Example:* a model that always says "not cancer" on a set with 98% healthy images gets 98% accuracy but terrible macro F1.

**Confusion matrix.** A table with true classes as rows and predicted classes as columns. The diagonal shows per class recall; off diagonal cells show which classes get confused.

**Calibration and ECE.** A model is calibrated if, among all the times it says "90% sure", it is right about 90% of the time, like a weather forecaster whose "70% chance of rain" days really get rain 70% of the time. **Expected calibration error (ECE)** groups predictions by confidence, measures the gap between confidence and accuracy in each group, and averages those gaps weighted by group size. *Example:* if half the predictions are at 75% confidence but only 50% right (gap 0.25), and half are at 55% confidence and all right (gap 0.45), ECE = 0.5 × 0.25 + 0.5 × 0.45 = 0.35.

**Reliability diagram.** A bar chart of accuracy in each confidence group. Perfect calibration puts every bar on the diagonal line.

**Temperature scaling.** Divide all logits by one number T before softmax. T above 1 softens overconfident predictions; T below 1 sharpens them. It never changes which class wins, so accuracy stays the same. T is fitted on validation data. *Lesson from this project:* it can only fix the miscalibration it can see on validation.

**Monte Carlo dropout.** Keep dropout switched on at test time and run the model many times (we used 30). Each run is like a slightly different model. If they agree, the model is sure; if they disagree, it is unsure. We average the 30 probability vectors.

**Entropy.** A measure of how spread out a probability distribution is. All probability on one class gives entropy 0 (certain); equal probability on all 9 classes gives the maximum, about 2.2 (completely unsure). *Example:* (0.98, 0.01, 0.01) has low entropy; (0.4, 0.3, 0.3) has high entropy.

**Accuracy versus coverage.** Let the model answer only its most confident X% of images and send the rest to a human. Coverage is X; we plot accuracy on the answered images. A good uncertainty score makes the curve rise steeply as coverage drops. *Example:* at 50% coverage our model is 96.9% accurate, versus 87.5% when it answers everything.

**Out of distribution (OOD) detection.** Spotting inputs unlike the training data: a blurry scan, a different stain, or something that is not tissue at all. We score every image with an uncertainty measure and check whether unfamiliar images score higher than normal ones.

**AUROC.** If you pick one normal image and one unfamiliar image at random, AUROC is the chance the unfamiliar one gets the higher uncertainty score. 0.5 means the score cannot tell them apart (a coin flip), 1.0 means perfect separation, and below 0.5 means it gets it backwards. *Example:* plain confidence scored 0.17 on blood cells, so it ranked blood cells as *more* familiar than real tissue.

**Grad CAM.** A heatmap of which image regions pushed the model toward a class. It takes the last convolution layer's 7 by 7 feature maps, weights each map by how much it increases the class score (the gradient), adds them up, keeps the positive part, and stretches the result to 28 by 28. It shows *where*, not *why*.

## 4. Major decisions and why

| Decision | Why |
|---|---|
| Python 3.12, not 3.11 or the installed 3.14 | Closest to Google Colab, and every package has stable builds for it. |
| Keep the official splits | A test set from another hospital is a more honest estimate of real use than a random split. |
| Report macro F1 and per class recall, not just accuracy | The test set has a different class mix, and some classes matter more clinically. |
| Logistic regression fitted with `L-BFGS` in PyTorch | scikit learn's solver ran for many minutes without converging on 2,352 features. Ours minimizes the same objective, which a test checks against scikit learn. |
| Choose logistic regression's C on validation, and extend the grid until the best value was not at the edge | A best value at the edge of the grid means the true best might lie beyond it. |
| Random flips and 90 degree rotations | Tissue has no orientation, so these are free, valid extra examples. |
| Early stopping on validation loss with patience 4 | Validation loss was noisy (spikes at epochs 5 and 12), so stopping at the first bad epoch would have stopped far too early. |
| Dropout only in the classifier head | Needed for Monte Carlo dropout, and it lets the convolution features be computed once and the cheap head rerun 30 times, giving identical results much faster. |
| ResNet18 on CPU with a 25 minute budget | Your plan was to try the CPU first. It ran, but only 4 epochs, so the README marks it as undertrained. |
| Memory fixes: uint8 storage, per batch normalization, chunked statistics, `num_workers=0` | The first training run was stopped for low memory. Peak RAM fell to 1.14 GB. |
| Commit the 0.46 MB small CNN checkpoint | Phases 3 and 4 can then be reproduced with exactly the reported model. |
| Fit the temperature on validation, never on test | Fitting on test would be peeking at the exam; and it revealed that an in house fix does not transfer. |
| Use three uncertainty scores (max softmax, softmax entropy, MC dropout entropy) | The plan asked to compare MC dropout against max softmax; entropy shows whether the gain comes from the score or from the dropout. |
| Corruptions at three severities, plus BloodMNIST | Shows both mild realistic problems and a completely different domain. |
| Grad CAM with two sanity checks | Heatmaps can look convincing even when meaningless; comparing against an untrained model tests that. |

## 5. Honest limitations and what I would do next

**Limitations.** 28 by 28 images lose fine detail. One dataset with one external test hospital. H&E stained images, not the label free infrared images used in the Bhargava lab. Unfamiliar data was simulated (corruptions) or from a different domain (blood), not real lab failures. ResNet18 was undertrained. One training run per model, so no error bars. Validation was used for several decisions and does not resemble the test hospital. The small CNN took 22 minutes to train, above the 15 minute target.

**Next steps.** Train ResNet18 properly on a Colab GPU at higher resolution. Add stain normalization or color augmentation and check whether the 10 point hospital gap shrinks. Try deep ensembles and dropout in the convolution layers. Add a dedicated "is this tissue?" check (for example, distance from training data in the model's feature space). Repeat runs with several seeds.

## 6. Fifteen questions a researcher might ask, with answers

1. **What is the main finding?** A small CNN classifies colon tissue well (87.5% at a new hospital), and Monte Carlo dropout can flag uncertain cases, raising accuracy to 96.9% on the half it keeps. But confidence alone is not trustworthy: the model is overconfident at the new hospital and confidently mislabels completely foreign images.

2. **Why does the model drop 10 points from validation to test?** Validation comes from the same source as training; test comes from another clinical center. Our reweighting analysis shows the color difference is a real staining or scanning shift, not just a different class mix.

3. **How do you know it is staining and not class mix?** I reweighted the training images to match the test set's exact class proportions. The color distance between train and test stayed the same (about 0.17 to 0.20 per channel), so class mix does not explain it.

4. **Why did temperature scaling not help?** It is fitted on validation data, where the model was already almost perfectly calibrated (ECE 0.003), so it had nothing to correct. The overconfidence only appears at the new hospital. A calibration fix learned in house cannot anticipate a shift it never sees.

5. **Why use Monte Carlo dropout instead of just the softmax probability?** Softmax confidence comes from one forward pass and is often overconfident. Running the model 30 times with random dropout shows whether its answer is stable. It separated right from wrong answers better (AUROC 0.83 versus 0.77) and detected noise much better (up to 0.96 versus 0.63 to 0.77).

6. **Why did nothing detect the blood cell images?** A softmax classifier must spread probability over its 9 tissue classes; it has no "none of the above". Blood smears are smooth and pale with a few purple blobs, so the model put them confidently into "background". Uncertainty methods built on the same classifier inherit this blind spot, which is why a separate check is needed.

7. **What is AUROC and why use it for out of distribution detection?** It is the probability that a random unfamiliar image gets a higher uncertainty score than a random normal one. It does not require choosing a threshold, and 0.5 means chance. Our 0.17 on blood cells means the score was confidently backwards.

8. **Why report macro F1?** Accuracy is dominated by common and easy classes. Macro F1 weighs all 9 classes equally. Logistic regression showed why: higher accuracy on test than validation but lower macro F1, because the test set had more easy adipose images.

9. **Why did ResNet18 do worse than your small CNN?** It is undertrained: only 4 epochs on a CPU versus 28 for the small CNN, and it was still improving. It was also designed for 224 by 224 photos, and our 28 by 28 inputs shrink to a single cell by its last block. The comparison is not fair yet, so I would not conclude that pretraining fails here.

10. **How do you know the Grad CAM heatmaps mean anything?** Two sanity checks. Heatmaps from the trained model are unrelated to heatmaps from an untrained copy (median correlation −0.02), so they reflect learning, not just image structure. And they change with the class being explained (−0.13 between the top two classes).

11. **What were the most dangerous errors?** 219 wrong predictions had over 99% confidence. The most confident ones were smooth muscle from the new hospital called background at 100%. These are invisible to any confidence threshold.

12. **How would this apply to the Bhargava lab's infrared imaging?** The images are different (chemical spectra, not stained color), but the questions transfer directly: is the model calibrated on new samples, can it hand off uncertain cases, and can it detect unfamiliar inputs? Label free imaging avoids stain variation, which might shrink the hospital gap we saw, but instrument and sample preparation shifts would still need these checks.

13. **How reproducible is this?** Seeds are fixed, package versions are pinned, every number is saved in `results/`, and run logs are committed. The small CNN checkpoint is committed, so Phases 3 and 4 give identical numbers. Retraining on another machine may differ slightly because CPU arithmetic is not perfectly deterministic.

14. **What would you do with more time or compute?** Train ResNet18 properly on a GPU, add stain augmentation to shrink the hospital gap, try deep ensembles, add a feature space "is this tissue?" detector, and repeat runs with several seeds to get error bars.

15. **What did you learn that surprised you?** That a model can be nearly perfectly calibrated in house and still overconfident elsewhere, and that it can be *more* confident on images it has never seen anything like than on real tissue. High confidence is a property of the model, not evidence that it is right.

## 7. Glossary

* **Adipose:** fat tissue.
* **Augmentation:** creating varied copies of training images (flips, rotations).
* **AUROC:** area under the ROC curve; chance a random positive outranks a random negative.
* **Batch normalization:** a layer that rescales values inside the network to keep training stable.
* **Calibration:** whether confidence matches accuracy.
* **Checkpoint:** a saved copy of a model's parameters.
* **CNN:** convolutional neural network.
* **Confusion matrix:** table of true versus predicted classes.
* **Cross entropy:** the loss used for classification; small when the right class gets high probability.
* **Dropout:** randomly switching off neurons during training.
* **ECE:** expected calibration error.
* **Entropy:** how spread out a probability distribution is.
* **Epoch:** one full pass through the training data.
* **Grad CAM:** gradient weighted class activation map; a heatmap of evidence.
* **H&E:** hematoxylin (purple, nuclei) and eosin (pink, cytoplasm) staining.
* **Logit:** a raw class score before softmax.
* **Macro F1:** the F1 score averaged equally over classes.
* **MC dropout:** Monte Carlo dropout; dropout kept on at test time over many runs.
* **Mini batch:** the small group of images used for one training step (here 128).
* **Out of distribution:** data unlike the training data.
* **Overfitting:** memorizing training data instead of learning general patterns.
* **Parameter:** one adjustable number inside a model.
* **Precision:** of the images predicted as a class, the share that truly are.
* **Recall:** of the images truly in a class, the share the model found.
* **Softmax:** turns logits into probabilities that sum to 1.
* **Stroma:** connective tissue; "cancer associated stroma" surrounds a tumor.
* **Temperature scaling:** dividing logits by T to adjust confidence.
* **Transfer learning:** starting from a model trained on another, larger dataset.
* **Validation set:** data held out from training and used to make decisions.
