# Deepfake Voice Detector - Project Process

**TL;DR**

- Task: decide whether a speech clip is real or AI-generated (fake), using the
  In-the-Wild dataset (31,779 clips, 54 speakers).
- Data is split by speaker (70% train / 30% validation), so the models are always
  tested on voices they never heard during training.
- Best model so far: **EfficientNet-B0** (F1 0.9852, ROC-AUC 0.9991, EER 1.255%).
- Main open problem: all models struggle with one validation speaker (Adam Driver);
  even the best model catches only 47.4% of his fake clips. His fakes are much louder
  in the original recordings and spectrally look more like real speech than like the
  training fakes (Step 9).

**Procedures** (status)

| # | Step | Status |
|---:|---|---|
| 0 | Environment setup and project layout | Done |
| 1 | Dataset download and organization | Done |
| 2 | Exploratory data analysis (EDA): counts, duration, sample rate, loudness, waveforms, Mel-spectrograms | Done |
| 3 | Preprocessing: fixed sample rate and duration, loudness normalization, log-Mel features | Done |
| 4 | Speaker-disjoint 70/30 split and DataLoaders | Done |
| 5 | Model A: simple CNN baseline | Done |
| 6 | Model B: transfer learning (pretrained ResNet18 / EfficientNet-B0) | Done |
| 7 | Model C: hybrid CNN + LSTM/GRU | Done |
| 8 | Evaluation and comparison: Accuracy, Precision/Recall, F1, confusion matrix, ROC-AUC, EER | Done |
| 9 | Error analysis and limitations | Done |

---

## Glossary (terms and abbreviations used in this document and in the figures)

**Data and audio**

| Term | Meaning |
|---|---|
| Real / bona-fide | A genuine human recording (label 0). |
| Fake / spoof | An AI-generated (deepfake) recording (label 1). The "positive" class in all metrics. |
| Speaker | The public figure whose voice is in the clip. |
| Speaker-disjoint split | Every speaker is in either training or validation, never both. |
| Sample rate (kHz) | Audio samples per second; all clips here are 16 kHz. |
| Mono | One audio channel. |
| RMS | Root mean square of the waveform: average signal energy, i.e. loudness. |
| dBFS | Decibels relative to full scale. 0 dBFS is the loudest possible level; more negative = quieter. |
| Peak level | The largest absolute sample value, in dBFS. |
| Clipping | Samples hitting the maximum level, which distorts audio. |
| Center crop | Keep the middle 4 seconds of a longer clip. |
| Wrap-pad (repeat-pad) | Make a short clip 4 seconds long by repeating it, instead of adding silence. |
| Shortcut | A simple cue (e.g. length, loudness) that predicts the label in this dataset but is not a real deepfake artifact. |

**Features**

| Term | Meaning |
|---|---|
| Spectrogram | A time x frequency picture of how much energy the audio has at each frequency over time. |
| Mel / Mel scale | A frequency scale that matches human hearing (finer at low frequencies). |
| log-Mel spectrogram | A Mel spectrogram in decibels (dB); the model input, shape 128 Mel bands x 126 time frames. |
| `n_mels`, `n_fft`, `hop_length` | Number of Mel bands; FFT window size in samples; step between frames in samples. |
| `fmin`, `fmax` | Lowest and highest frequency included (20 Hz - 8 kHz). |
| `top_db` | Values more than 80 dB below the clip's maximum are floored at -80 dB. |

**Statistics (EDA)**

| Term | Meaning |
|---|---|
| Median / mean | Middle value / average. |
| Mann-Whitney U test | Tests whether one group's values tend to be larger than the other's. |
| KS (Kolmogorov-Smirnov) statistic | Largest gap between two distributions (0 = identical, 1 = completely separate). |
| Cliff's delta | Effect size from -1 to 1: how often a value from one group is larger than one from the other (0 = no difference). |
| Wilcoxon signed-rank test | Paired test, here comparing real vs fake within the same speaker. |
| p-value | Probability of seeing a difference this large by chance; below 0.05 is usually called significant. |
| Within-speaker | Computed after removing each speaker's own typical value, so differences between speakers do not count. |

**Models and training**

| Term | Meaning |
|---|---|
| CNN | Convolutional neural network; learns local patterns in the spectrogram image. |
| Conv2d / BatchNorm / ReLU / MaxPool | Convolution layer / normalization layer / activation function / down-sampling layer. |
| Dropout | Randomly zeroes units during training to reduce overfitting. |
| Transfer learning | Start from a model pretrained on another task (here ImageNet images) and fine-tune it. |
| ImageNet | A large photo dataset used to pretrain ResNet18 and EfficientNet-B0. |
| ResNet18 / EfficientNet-B0 | Two standard pretrained image-classification networks. |
| RNN, GRU, LSTM | Recurrent networks that read a sequence step by step; GRU and LSTM are two common types. |
| BiGRU / BiLSTM | Bidirectional GRU / LSTM: read the sequence forwards and backwards. |
| Attention pooling | Learns a weight for each time step and averages them into one vector. |
| SpecAugment | Training-time augmentation that blanks random time and frequency bands. |
| Class weights | Make mistakes on the smaller class (fake) cost more in the loss. |
| CrossEntropyLoss | The standard classification loss. |
| AdamW | The optimizer (Adam with weight decay). |
| lr | Learning rate. |
| ReduceLROnPlateau | Halves the learning rate when validation ROC-AUC stops improving. |
| Epoch | One full pass over the training data. |
| Early stopping | Stop training when validation ROC-AUC has not improved for a set number of epochs (patience). |
| Params | Number of trainable parameters (M = million). |

**Evaluation metrics** (fake is the positive class; threshold 0.5 unless stated)

| Term | Meaning |
|---|---|
| TP / FP / TN / FN | True positive (fake caught) / false positive (real flagged as fake) / true negative (real accepted) / false negative (fake missed). |
| Confusion matrix | 2 x 2 table of TP, FP, TN, FN. Rows = true label, columns = predicted label. |
| Accuracy | Share of all clips classified correctly. |
| Precision | Of clips flagged as fake, the share that really are fake. |
| Recall (TPR) | Of the fake clips, the share that are caught. |
| Specificity (TNR) | Of the real clips, the share correctly accepted. |
| F1 | Harmonic mean of Precision and Recall. |
| FPR | False positive rate = 1 - Specificity. |
| FNR | False negative rate = 1 - Recall (share of fakes missed). |
| ROC curve | TPR vs FPR as the decision threshold moves from 1 to 0. |
| ROC-AUC (AUC) | Area under the ROC curve: probability that a random fake clip gets a higher fake score than a random real clip. 0.5 = chance, 1 = perfect. Values below 0.5 mean the cue works in the opposite direction. |
| EER | Equal error rate: the error rate at the threshold where FPR = FNR. Lower is better. |
| EER threshold | The fake-probability threshold where FPR = FNR. |
| DET-style curve | FNR plotted against FPR; the EER is where the curve crosses the diagonal. |
| Probability of fake | The model's softmax output for the fake class. |

---

## Key concepts

**Why can accuracy be misleading with a 63/37 split?**

If 63% of the data is real, a model that always predicts "real" already achieves
63% accuracy while catching zero fake clips. In an anti-fraud setting, missed fake
clips are costly. Precision, Recall, F1, the confusion matrix, and ROC-AUC are
therefore reported alongside accuracy.

**Why split by speaker instead of by file?**

A random file-level split can place the same speaker in both training and
validation. The model may then memorize the speaker's voice or recording channel
instead of learning spoofing artifacts, inflating validation performance. A
speaker-disjoint split simulates deployment on unseen voices and gives a more
honest estimate of generalization.

---

## Project overview

- Dataset: [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) on Hugging Face
- Dataset paper: Müller et al., *Does Audio Deepfake Detection Generalize?* (2022)
- License: CC BY-SA 4.0
- Features: log-Mel spectrograms extracted with librosa
- Models to compare: a simple CNN baseline, pretrained ResNet/EfficientNet transfer learning, and CNN + LSTM/GRU hybrids
- Hardware: WSL2 on a local workstation with an NVIDIA RTX 3060 Laptop GPU (6 GB)

---

## Step 0: Environment check and project layout

**Goal:** confirm that the local machine can run the project and create a reproducible
project structure.

**Environment**

- OS: WSL2 (Linux), Python 3.12.3
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), CUDA driver available
- RAM: 7.6 GB; free disk space: about 204 GB
- Project folders: `data/raw`, `data/processed`, `data/splits`, `outputs/{figures,models,logs}`, `src/`
- Isolated environment: project-local `.venv/` (system Python was not modified)
- Key versions: torch 2.15.0+cu132, torchvision 0.29.2, torchaudio 2.11.1, librosa 1.0.0, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1

**Verification**

- `torch.cuda.is_available() == True`; the RTX 3060 Laptop GPU is detected
- `src/00_environment_check.py` repeats the Python, package, GPU, and disk checks
- Two extra one-off manual checks (not part of the script): a 1000 x 1000 matrix
  multiplication on the GPU completed successfully, and librosa extracted a 128-band
  Mel-spectrogram from a 2-second, 16 kHz clip (shape `(128, 63)`)

---

## Step 1: Dataset download and organization

**Goal:** obtain all audio files, verify labels and speaker metadata, and organize
the data into a structure that later steps can read directly.

**Source and process**

- Dataset: [Hugging Face `mueller91/In-The-Wild`](https://huggingface.co/datasets/mueller91/In-The-Wild)
- License: CC BY-SA 4.0
- The dataset is distributed as `release_in_the_wild.zip`; the extracted files were placed in
  `data/raw/release_in_the_wild/`
- `meta.csv` was used to verify that every audio file has a matching metadata row.
  There were no missing files, no extra files, and no zero-byte files.
- Audio files were moved into `real/` (bona-fide) and `fake/` (spoof) subfolders.
  The move was done on the same disk, so no extra copy was created.
- An index table was written to `data/raw/dataset_index.csv` with the columns
  `file, speaker, label, label_id, path, size_bytes`.

**Key numbers**

| Item | Value |
|---|---:|
| Total audio clips | 31,779 |
| Total size | 8.12 GiB |
| Real (`real`) | 19,963 (62.8%) |
| Fake (`fake`) | 11,816 (37.2%) |
| Real : fake ratio | about 1.69 : 1 |
| Speakers in `meta.csv` | 54 (the dataset card lists 58) |

**Caveats**

- The classes are not strictly balanced. The Step 4 split therefore keeps the fake
  share of each split close to the global ratio, and evaluation does not rely on
  accuracy alone.
- Speaker labels are available. The Step 4 split is therefore speaker-disjoint, so
  the models are tested on voices they have never seen.
- Full duration, sample-rate, and loudness statistics are covered in Step 2.

**Takeaway**

> The dataset contains 31,779 real and fake speech clips, with roughly 1.69 times
> more real than fake samples. This is not an extreme imbalance, but it is enough
> for accuracy to be misleading; a class-balanced split and Recall/F1 metrics are
> therefore necessary. Speaker annotations also make it possible to build a
> speaker-disjoint validation set and test whether the model learns spoofing cues
> rather than speaker identity.

---

## Step 2: Exploratory data analysis (EDA)

**Goal:** characterize duration, sample rate, channels, and loudness; test for
systematic real/fake differences; and visualize waveforms and Mel-spectrograms.

**Method**

- Read all 31,779 clips (header plus full decode) and computed duration, RMS
  loudness in dBFS, peak level, and clipping ratio.
- Statistical tests: Mann-Whitney U, Kolmogorov-Smirnov, Cliff's delta; Wilcoxon
  signed-rank for paired within-speaker comparisons.
- Shortcut analysis: ROC-AUC of duration-only and loudness-only classifiers,
  including within-speaker variants and per-speaker AUC.
- Visualization sample: one real and one fake clip from the same speaker
  (Alec Guinness), each 3.5-4.5 seconds long and the closest to the median loudness
  of that speaker's clips of the same class and length range.
- Mean Mel: 120 randomly sampled clips per class (seed 42). Each clip is cut to its
  first 4 seconds, or zero-padded at the end if shorter; each clip's mean energy was
  subtracted before averaging so that loudness would not dominate the comparison.
- Outputs: `data/processed/audio_stats.csv`, `eda_label_summary.csv`,
  `eda_tests.csv`, `eda_speaker_summary.csv`, `eda_speaker_tests.csv`,
  `eda_shortcut_auc.csv`, and `eda_per_speaker_auc.csv`; a text summary in
  `outputs/eda_summary.txt`.

**Key numbers**

| Metric | Real | Fake |
|---|---:|---:|
| Clips | 19,963 | 11,816 |
| Total hours | 20.76 | 17.10 |
| Median duration | 2.99 s | 4.05 s |
| Mean duration | 3.74 s | 5.21 s |
| Median RMS loudness | -26.1 dBFS | -28.5 dBFS |
| Median peak level | -9.3 dBFS | -10.1 dBFS |

- Format: all 31,779 clips are 16 kHz, mono, and unclipped (clipping ratio approximately 0).
- Duration difference: KS = 0.188, Cliff's delta = -0.239.
- Within-speaker duration: 37 of 54 speakers have longer fake clips; median difference
  is +1.06 s; Wilcoxon p = 0.0022.
- Within-speaker loudness: median difference is -2.14 dB (fake quieter); Wilcoxon
  p = 0.096, and the direction is not consistent across speakers (21 of 54 speakers
  have louder fake clips).
- Single-feature AUC (0.5 = chance): duration 0.619, loudness 0.649,
  within-speaker duration 0.613, within-speaker loudness 0.706.
- Per-speaker AUC (39 speakers with at least 20 clips per class): median duration
  AUC = 0.596, median loudness AUC = 0.744. Loudness AUC is above 0.5 for 29 of 39
  speakers (fake is quieter) and below 0.5 for 10 (fake is louder). Both an AUC
  near 1 and an AUC near 0 mean loudness separates the classes well for that
  speaker; only the direction differs.

**Figures**

| Figure | What it shows |
|---|---|
| `outputs/figures/02_duration_distribution.png` | Fake clips are shifted toward longer durations. |
| `outputs/figures/02_loudness_distribution.png` | Fake clips are shifted toward lower RMS loudness. |
| `outputs/figures/02_speaker_class_composition.png` | Class ratios vary strongly by speaker; for example, Barack Obama has 3,302 real and 334 fake clips, while Alec Guinness has 1,907 real and 1,718 fake clips. |
| `outputs/figures/02_speaker_duration_scatter.png` | Most speakers lie above the y = x line: fake clips are usually longer even within the same speaker. |
| `outputs/figures/02_shortcut_auc_by_speaker.png` | Loudness is a strong shortcut for many speakers, with AUC values close to 0 or 1. |
| `outputs/figures/02_waveform_comparison.png` | Waveforms from the same speaker look different in amplitude and envelope. |
| `outputs/figures/02_mel_comparison.png` | The fake example has a much darker background (lower noise floor) and less high-frequency energy; this is illustrative only. |
| `outputs/figures/02_mean_mel_by_class.png` | After removing loudness, the average spectra still differ, with real speech showing relatively more energy in the upper bands. The difference map after about 2 seconds mostly reflects zero padding (real clips are shorter), not a spectral difference. |

**Caveats**

- The dataset contains clear shortcuts:
  - Fake clips are longer overall, and also tend to be longer within the same speaker.
  - Fake clips are quieter overall. Per-speaker loudness AUC has a median of 0.744,
    making loudness the strongest shortcut found in this dataset.
- Speaker composition is strongly correlated with class balance. A random file-level
  split would leak both speaker identity and the duration/loudness shortcuts.
- The single waveform and Mel examples are illustrative, not causal evidence. The
  averaged Mel comparison is more reliable but still descriptive, and it is partly
  confounded by clip length: this EDA figure zero-pads short clips, and real clips
  are shorter, so the later time frames mix in silence. The upper-band difference
  itself is confirmed in Step 9 on the wrap-padded model inputs (training real
  -11.9 dB vs training fake -13.5 dB relative high-band energy).
- Step 3 therefore standardizes duration (crop/pad) and normalizes loudness per
  clip. A non-normalized control version was planned to measure how much
  performance depends on shortcuts, but it was not run (see Step 9 Limitations).
- Step 4 therefore uses a speaker-disjoint split.

**Takeaway**

> EDA revealed two shortcuts that are easier to learn than the actual spoofing
> artifacts: fake clips are about 1.06 seconds longer within the same speaker
> (p = 0.002), and they are about 2.4 dB quieter overall (median RMS). Loudness
> alone reaches a median per-speaker AUC of 0.744. Without loudness normalization and duration
> standardization, a CNN may become a "volume/length detector" that fails when the
> speaker or recording condition changes. This is also why a speaker-disjoint
> validation set is essential.

---

## Step 3: Preprocessing and feature extraction

**Goal:** standardize sample rate, duration, and loudness; extract log-Mel
spectrograms; and save a feature matrix that later training steps can load
directly.

**Pipeline**

- Sample rate: resample every clip to 16 kHz mono (the dataset already satisfies this).
- Duration: fix every clip to 4 seconds. Long clips are center-cropped; short clips
  are repeat-padded (wrap-pad) instead of zero-padded, avoiding a silence shortcut.
- Loudness: normalize each clip to -23 dBFS RMS. If the peak exceeds 0.99, scale the
  clip down to avoid clipping.
- Mel parameters: `n_mels=128`, `n_fft=2048`, `hop_length=512`, `fmin=20`,
  `fmax=8000`, `power_to_db(ref=np.max, top_db=80)`.
- Parallelism: run with `--workers 12` (script default is 8); the full run over
  31,779 clips took about 24 minutes.
- Output directory: `data/processed/mel/`.

**Key numbers**

| Item | Value |
|---|---:|
| Feature matrix | `mel_spectrograms.npy` |
| Shape | `(31779, 128, 126)` |
| Data type | float32 |
| File size | about 2.0 GB |
| Labels | 0 = Real (19,963); 1 = Fake (11,816) |
| Value range | -80 to 0 dB |
| Invalid values | no NaN, no Inf |
| Metadata rows | 31,779 (no read errors) |

- Median RMS before normalization: Real -25.9 dBFS, Fake -28.5 dBFS. These are
  measured on the 4-second cropped/padded clip, so they differ slightly from the
  full-clip values in Step 2 (-26.1 / -28.5 dBFS).
- Median RMS after normalization: -23.0 dBFS for both classes.
- Loudness-only shortcut AUC: 0.649 before normalization, 0.505 after.
- 61.0% of clips are shorter than 4 seconds and require wrap-padding (Real 67.8%,
  Fake 49.4%).
- Median padding: Real 1.01 s, Fake 0.00 s. These numbers are written to
  `data/processed/mel/preprocessing_summary.txt`.
- Residual padding shortcut: AUC = 0.399 (0.5 means no class signal).
- Mean center-crop start: Real 0.43 s, Fake 0.99 s (this value is not a model input).

**Figures**

| Figure | What it shows |
|---|---|
| `outputs/figures/03_loudness_before_after.png` | RMS distributions before and after normalization; after normalization both classes overlap at -23 dBFS. |
| `outputs/figures/03_preprocessed_examples.png` | Final waveform and log-Mel input for one real and one fake clip from the same speaker. |
| `outputs/figures/03_padding_by_class.png` | Wrap-padding needed to reach 4 seconds; the residual padding AUC is 0.399. |

**Caveats**

- The strong loudness shortcut is essentially removed, but 61% of clips still need
  repeat-padding. The padding amount retains a weak class signal, and a model could
  potentially exploit repeated content.
- Training therefore uses SpecAugment (Steps 5-7), and Step 9 groups mistakes by
  padding amount. That analysis found no padding effect except on clips shorter
  than 1 second.
- Feature mean and standard deviation for neural networks must be computed on the
  training split only, never on the full dataset, to avoid validation leakage.
- A non-normalized control feature set can be generated with
  `--no-normalize --output-dir data/processed/mel_raw` for an ablation study; this
  ablation has not been run.
- `data/processed/` is ignored by git, so generated features are not committed.

**Takeaway**

> Per-clip RMS normalization reduced the loudness-only AUC from 0.649 to 0.505,
> essentially removing the strongest shortcut found in EDA. Fixing the duration at
> 4 seconds also removed the direct duration cue. However, 61% of clips are shorter
> than 4 seconds and require repeat-padding, and the padding amount still carries a
> weak class signal (AUC = 0.399). Preprocessing alone is therefore not enough:
> the later steps add training-time augmentation (Steps 5-7), a speaker-disjoint
> split (Step 4), and error analysis grouped by padding amount (Step 9) to test
> whether the models learn genuine spoofing artifacts.

---

## Step 4: Dataset split (70/30) and DataLoaders

**Goal:** create a speaker-disjoint training/validation split, compute normalization
statistics on the training split only, and build PyTorch Dataset/DataLoader objects.

**Method**

- Split at the speaker level: every clip from a speaker goes to either training or
  validation, never both.
- Fixed random seed 42. Search 5,000 random speaker permutations and choose the one
  that keeps:
  - validation close to 30% of all clips;
  - validation fake share close to the global 37.18%;
  - both classes present in both splits.
- Normalization: compute the global feature mean/std on the training split only;
  validation reuses those statistics.
- Class weights: inverse-frequency CrossEntropy weights computed from the training
  split to reduce the effect of the 63/37 imbalance.
- Dataset: lazily reads rows from the `.npy` memmap and returns a `(1, 128, 126)`
  float32 tensor plus a `long` label.
- DataLoader: batch size 32, shuffled training loader, unshuffled validation loader,
  4 workers, and pinned memory when CUDA is available.

**Key numbers**

| Item | Train | Validation |
|---|---:|---:|
| Clips | 22,233 (69.96%) | 9,546 (30.04%) |
| Speakers | 40 | 14 |
| Real clips | 13,954 | 6,009 |
| Fake clips | 8,279 | 3,537 |
| Fake share | 37.24% | 37.05% |

- Global fake share: 37.18%.
- Speaker overlap: 0; row-index overlap: 0.
- Class weights (Real/Fake): `[0.7967, 1.3427]`.
- Training feature mean/std: `-50.6645 / 17.4524`.
- DataLoader: 695 training batches and 299 validation batches; batch shape
  `(32, 1, 128, 126)`.
- After standardization, the training batch has mean/std about `-0.014 / 0.956`;
  the validation batch has about `-0.107 / 1.039`.
- Validation speakers: 2Pac, Adam Driver, Alan Watts, Ayn Rand, Barack Obama,
  Bob Ross, Dwight Eisenhower, Kamala Harris, Lyndon Johnson, Mr. Rogers,
  Norm MacDonald, Ronald Reagan, The Notorious B.I.G., and William F. Buckley.

**Figure**

- `outputs/figures/04_split_composition.png`: real/fake counts per split and fake
  share versus the global 37.2% line.

**Caveats**

- The validation set contains only 14 speakers, so results can vary depending on
  which speakers happen to land in validation. Speaker-level cross-validation would
  provide a more robust estimate later.
- Overall class balance is preserved, but individual speakers have very different
  real/fake ratios (for example, Barack Obama has far more real than fake clips).
  The validation set may therefore be harder than a random file-level split.
- Normalization statistics come from the training split. The validation batch mean
  is slightly shifted (-0.107), which reflects the different speaker distribution
  and should not be corrected using validation data.
- This Dataset does not apply augmentation. The training scripts (Steps 5-7) add
  SpecAugment and use the saved class weights.

**Takeaway**

> The speaker-disjoint 70/30 split produced 22,233 training clips and 9,546
> validation clips with zero speaker overlap, while preserving the class ratio
> (37.24% fake in training, 37.05% in validation, 37.18% globally). This is stricter
> than a random file-level split and tests generalization to unseen speakers. The
> DataLoader reliably yields `(32, 1, 128, 126)` tensors, and normalization
> statistics are computed on the training split only, avoiding validation leakage.

---

## Step 5: Model A - simple CNN baseline

**Goal:** train a small, reproducible CNN on the speaker-disjoint split and establish
a baseline for the later transfer-learning and hybrid models.

**Architecture**

- Input: normalized log-Mel of shape `(1, 128, 126)`.
- Four convolution stages: Conv2d(1->32), Conv2d(32->64), Conv2d(64->128),
  Conv2d(128->256).
- Each stage uses a 3x3 convolution, BatchNorm, and ReLU; the first three stages
  use 2x2 max pooling.
- Head: AdaptiveAvgPool2d(1), Dropout(0.4), Linear(256->2).
- Trainable parameters: 388,834.

**Training setup**

- Loss: CrossEntropyLoss with training-only class weights `[0.7967, 1.3427]`.
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4.
- Scheduler: ReduceLROnPlateau on validation ROC-AUC, factor=0.5, patience=2.
- Augmentation: training-time SpecAugment (2 time masks and 2 frequency masks,
  maximum width 10); no augmentation during validation.
- Batch size 32, up to 20 epochs, early stopping patience 5 on validation ROC-AUC.
- Seed 42; training took about 5.4 minutes on the RTX 3060 Laptop GPU.
- Best epoch: 12; training stopped after epoch 17.

**Validation results**

| Metric | Value |
|---|---:|
| Accuracy | 0.9852 |
| Precision | 0.9994 |
| Recall | 0.9607 |
| F1 | 0.9797 |
| ROC-AUC | 0.9957 |

- Confusion matrix: `[[6007, 2], [139, 3398]]`.
- False positives (real predicted as fake): 2.
- False negatives (fake predicted as real): 139.
- Mean predicted probability of fake: 0.0004 for real clips, 0.9599 for fake clips.
- **Key anomaly:** 134 of the 139 false negatives come from a single validation
  speaker, Adam Driver. The aggregate score hides a severe speaker-level failure.
- The per-speaker numbers and mean probabilities are produced by
  `src/08_evaluate_models.py` (`outputs/logs/08_per_speaker_breakdown.csv`).

**Figures**

- `outputs/figures/05_cnn_training_curves.png`: training/validation loss and validation
  F1/ROC-AUC curves.
- `outputs/figures/05_cnn_confusion_matrix.png`: validation confusion matrix with row
  percentages.
- `outputs/figures/05_cnn_roc_curve.png`: validation ROC curve.
- `outputs/figures/05_cnn_probability_distribution.png`: predicted fake-probability
  distributions for real and fake clips.

**Outputs**

- `outputs/models/05_cnn_baseline.pt`: best state dict plus feature mean/std and class
  weights.
- `outputs/logs/05_cnn_history.csv`: per-epoch training log.
- `outputs/logs/05_cnn_metrics.json`: full metrics for the best epoch.
- `outputs/logs/05_cnn_val_predictions.csv`: per-clip validation probabilities for
  Step 9 error analysis.

**Caveats**

- A single speaker-disjoint split contains only 14 validation speakers, so the
  metrics have substantial variance. One split is not enough to claim robust
  generalization.
- The Adam Driver failures show that aggregate Accuracy/AUC can hide speaker-level
  collapse. Step 9 breaks errors down by speaker.
- The high score may partly reflect dataset-specific generation or recording
  artifacts; cross-dataset generalization is not guaranteed (see Step 9 Limitations).
- The threshold is fixed at 0.5. In anti-fraud use, missed fake clips matter more;
  Step 9 analyzes the recall-threshold trade-off.

**Takeaway**

> A simple CNN with only 388,834 parameters already reaches 98.52% accuracy, 97.97%
> F1, and 99.57% ROC-AUC on unseen speakers, making log-Mel + CNN a strong baseline
> for this dataset. However, 134 of its 139 missed fake clips come from one speaker,
> which shows why aggregate metrics are not enough: later models must be compared on
> hard-speaker recall and cross-domain robustness, not only on average scores.

---

## Step 6: Model B - pretrained ResNet / EfficientNet transfer learning

**Goal:** compare ImageNet-pretrained ResNet18 and EfficientNet-B0 on the same
speaker-disjoint split and test whether pretrained features improve on the simple
CNN baseline.

**Input and preprocessing**

- Map saved log-Mel dB values from `[-80, 0]` to `[0, 1]`.
- Resize bilinearly to `224 x 224`.
- Repeat the single channel to three channels to match ImageNet pretrained inputs.
- Normalize with ImageNet mean/std. During training, apply SpecAugment after
  normalization (2 time masks and 2 frequency masks, maximum width 16).
- No augmentation during validation.

**Models and training**

- ResNet18: ImageNet pretrained, replace `fc` with a two-class head; 11,177,538
  trainable parameters.
- EfficientNet-B0: ImageNet pretrained, replace `classifier[1]`; 4,010,110
  trainable parameters.
- Differential learning rates: backbone 1e-4, head 1e-3.
- AdamW, weight_decay=1e-4; CrossEntropyLoss with training class weights.
- ReduceLROnPlateau on validation ROC-AUC (factor=0.5, patience=2).
- Batch size 32, up to 12 epochs, early stopping patience 4.
- ResNet18: best epoch 10, ran all 12 epochs (no early stop), about 20.6 minutes.
- EfficientNet-B0: best epoch 5, stopped at epoch 9, about 33.1 minutes.

**Validation comparison**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Simple CNN | 0.9852 | 0.9994 | 0.9607 | 0.9797 | 0.9957 | 0.39M | 5.4 min |
| ResNet18 | 0.9872 | 0.9980 | 0.9675 | 0.9825 | 0.9975 | 11.18M | 20.6 min |
| EfficientNet-B0 | **0.9891** | 0.9911 | **0.9794** | **0.9852** | **0.9991** | 4.01M | 33.1 min |

- ResNet18 confusion matrix: `[[6002, 7], [115, 3422]]`.
- EfficientNet-B0 confusion matrix: `[[5978, 31], [73, 3464]]`.
- Total errors: Simple CNN 141 -> ResNet18 122 -> EfficientNet-B0 104.
- False negatives (fake predicted as real): Simple CNN 139 -> ResNet18 115 ->
  EfficientNet-B0 73.
- False positives (real predicted as fake): Simple CNN 2 -> ResNet18 7 ->
  EfficientNet-B0 31.
- Adam Driver fake recall: Simple CNN **2.2%** -> ResNet18 **16.8%** ->
  EfficientNet-B0 **47.4%** (from `outputs/logs/08_per_speaker_breakdown.csv`).
- EfficientNet-B0 is best in Accuracy/F1/AUC/Recall; ResNet18 has slightly higher
  Precision. Both trade more false positives for higher recall.

**Figures**

- ResNet18:
  - `outputs/figures/06_resnet18_training_curves.png`
  - `outputs/figures/06_resnet18_confusion_matrix.png`
  - `outputs/figures/06_resnet18_roc_curve.png`
  - `outputs/figures/06_resnet18_probability_distribution.png`
- EfficientNet-B0:
  - `outputs/figures/06_efficientnet_b0_training_curves.png`
  - `outputs/figures/06_efficientnet_b0_confusion_matrix.png`
  - `outputs/figures/06_efficientnet_b0_roc_curve.png`
  - `outputs/figures/06_efficientnet_b0_probability_distribution.png`

**Outputs**

- `outputs/models/06_resnet18_best.pt`
- `outputs/models/06_efficientnet_b0_best.pt`
- `outputs/logs/06_*_history.csv`
- `outputs/logs/06_*_metrics.json`
- `outputs/logs/06_*_val_predictions.csv`

**Caveats**

- Transfer learning improves on the simple CNN, but the gain is modest
  (F1 0.9797 -> 0.9852), suggesting that the log-Mel patterns in this dataset are
  already easy to learn.
- The pretrained weights come from ImageNet image classification, not audio.
  Repeating the channel to three channels is only an interface adaptation; it does
  not mean the model understands spectrograms in the same way as images.
- EfficientNet-B0 has fewer parameters but trains more slowly (33.1 min vs 20.6 min)
  because of depthwise convolutions and the deeper architecture at 224 x 224.
- Adam Driver remains the hardest speaker: even EfficientNet-B0 reaches only 47.4%
  fake recall and still misses 72 fake clips.
- The single speaker-disjoint split contains only 14 validation speakers, so one
  result is not enough to claim that a transfer model is more robust.

**Takeaway**

> Pretrained transfer learning raises EfficientNet-B0's F1 from 0.9797 (simple CNN)
> to 0.9852 and ROC-AUC from 0.9957 to 0.9991, while increasing fake recall on the
> hardest speaker, Adam Driver, from 2.2% to 47.4%. The gain is real but limited, and
> EfficientNet-B0 pays for it with more false positives (31 vs 2). Pretraining helps,
> but replacing the model alone does not solve hard speakers or cross-domain
> robustness; those remain the focus of the next steps.

---

## Step 7: Model C - CNN + LSTM/GRU hybrid models

**Goal:** explicitly model temporal dynamics and test whether CNN+BiGRU or
CNN+BiLSTM can beat the simple CNN and the transfer-learning models.

**Architecture**

- CNN front-end: three convolution blocks with 32 -> 64 -> 128 channels; each block
  is 3x3 Conv + BatchNorm + ReLU, followed by Dropout2d(0.1) after the last block.
- Pooling: 128x126 -> 64x63 -> 32x31 -> 16x31 (frequency reduced to 16 while
  31 time steps are kept).
- Sequence conversion: flatten the 16x128 frequency/channel grid into a 2048-dim
  feature vector per time step.
- Temporal layer: bidirectional GRU or bidirectional LSTM, hidden size 128.
- Aggregation: additive attention over the 31 time steps.
- Classifier: Dropout(0.3) + Linear(256->2).
- Trainable parameters: CNN+BiGRU 1,766,371; CNN+BiLSTM 2,323,939.

**Training setup**

- Input: `(1, 128, 126)` log-Mel normalized with training-set mean/std.
- Loss: CrossEntropyLoss with training class weights.
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4.
- Gradient clipping: max_norm=5.0.
- Scheduler: ReduceLROnPlateau on validation ROC-AUC (factor=0.5, patience=2).
- Batch size 32, up to 20 epochs, early stopping patience 5.
- CNN+BiGRU: best epoch 5, stopped at epoch 10, about 3.0 minutes.
- CNN+BiLSTM: best epoch 10, stopped at epoch 15, about 6.5 minutes.

**Validation comparison** (new hybrids vs the baseline and the best model so far;
the full five-model table is in Step 8)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Errors | Adam Driver fake recall | Params | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple CNN | 0.9852 | **0.9994** | 0.9607 | 0.9797 | 0.9957 | 141 | 2.2% | 0.39M | 5.4 min |
| EfficientNet-B0 | **0.9891** | 0.9911 | **0.9794** | **0.9852** | **0.9991** | **104** | **47.4%** | 4.01M | 33.1 min |
| CNN+BiGRU | 0.9848 | 0.9953 | 0.9635 | 0.9792 | 0.9889 | 145 | 8.0% | 1.77M | 3.0 min |
| CNN+BiLSTM | 0.9854 | 0.9948 | 0.9658 | 0.9801 | 0.9916 | 139 | 12.4% | 2.32M | 6.5 min |

- CNN+BiGRU confusion matrix: `[[5993, 16], [129, 3408]]`.
- CNN+BiLSTM confusion matrix: `[[5991, 18], [121, 3416]]`.
- CNN+BiGRU: 129 false negatives and 16 false positives; CNN+BiLSTM: 121 false
  negatives and 18 false positives.
- BiLSTM's F1 is slightly higher than the simple CNN (0.9801 vs 0.9797), but its
  ROC-AUC is lower (0.9916 vs 0.9957).
- Both hybrid models have lower AUC than EfficientNet-B0, and much lower Adam Driver
  fake recall than EfficientNet-B0's 47.4%.

**Figures**

- CNN+BiGRU:
  - `outputs/figures/07_cnn_bigru_training_curves.png`
  - `outputs/figures/07_cnn_bigru_confusion_matrix.png`
  - `outputs/figures/07_cnn_bigru_roc_curve.png`
  - `outputs/figures/07_cnn_bigru_probability_distribution.png`
- CNN+BiLSTM:
  - `outputs/figures/07_cnn_bilstm_training_curves.png`
  - `outputs/figures/07_cnn_bilstm_confusion_matrix.png`
  - `outputs/figures/07_cnn_bilstm_roc_curve.png`
  - `outputs/figures/07_cnn_bilstm_probability_distribution.png`

**Outputs**

- `outputs/models/07_cnn_bigru_best.pt`
- `outputs/models/07_cnn_bilstm_best.pt`
- `outputs/logs/07_cnn_*_history.csv`
- `outputs/logs/07_cnn_*_metrics.json`
- `outputs/logs/07_cnn_*_val_predictions.csv`

**Caveats**

- The hybrid structure did not beat transfer learning. Flattening 16 frequency bins
  x 128 channels into a 2048-dim sequence input may give the RNN too many parameters
  and make it easier to overfit.
- There are only 31 time steps, so the available temporal context is limited. The
  spoofing artifacts in this dataset may be more local/frame-level, so explicit
  sequence modeling does not necessarily help.
- Only one hybrid design was tried. Hyperparameters such as hidden size, pooling,
  depth, and learning rate were not exhaustively tuned; this result does not prove
  that all CNN+RNN models are ineffective.
- Adam Driver remains hard: BiGRU/BiLSTM fake recall is only 8.0%/12.4%, far below
  EfficientNet-B0.

**Takeaway**

> CNN+BiLSTM slightly improves F1 over the simple CNN (0.9801 vs 0.9797) but lowers
> ROC-AUC (0.9916 vs 0.9957); CNN+BiGRU is lower on both. Both hybrids are clearly
> below EfficientNet-B0's 0.9991 AUC, showing that a more complex temporal model is
> not automatically better. One likely reason is that flattening the frequency axis
> creates a very high-dimensional, short sequence, so the RNN has little temporal
> context to exploit; another is that the dataset's artifacts are mostly local. This
> negative result is useful: the next focus should be hard speakers, data design, and
> evaluation, not simply adding model complexity.

---

## Step 8: Model evaluation and comparison (including EER)

**Goal:** compare all five models on the same speaker-disjoint validation split using
Accuracy, Precision, Recall, F1, ROC-AUC, and Equal Error Rate (EER).

**Evaluation setup**

- Validation set: 9,546 clips from 14 unseen speakers.
- Classification threshold: 0.5 for Accuracy/Precision/Recall/F1 and confusion
  matrices.
- EER: the error rate where FPR equals FNR; lower is better. The probability
  threshold closest to the EER operating point is also reported.
- Prediction probabilities come from each model's saved validation predictions. The
  script checks that all five models use identical row indices and labels.
- Caveat: the validation split was also used for early stopping and model selection,
  so these numbers are mildly optimistic and are not an independent test set.

**Comparison table**

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | EER (%) | EER threshold | EER rank | Adam Driver fake recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | **0.9891** | 0.9911 | **0.9794** | 0.9948 | **0.9852** | **0.9991** | **1.255** | 0.1491 | 1 | **47.4%** |
| ResNet18 | 0.9872 | 0.9980 | 0.9675 | 0.9988 | 0.9825 | 0.9975 | 2.184 | 0.0055 | 2 | 16.8% |
| Simple CNN | 0.9852 | **0.9994** | 0.9607 | **0.9997** | 0.9797 | 0.9957 | 2.393 | 0.0012 | 3 | 2.2% |
| CNN+BiLSTM | 0.9854 | 0.9948 | 0.9658 | 0.9970 | 0.9801 | 0.9916 | 2.834 | 0.0164 | 4 | 12.4% |
| CNN+BiGRU | 0.9848 | 0.9953 | 0.9635 | 0.9973 | 0.9792 | 0.9889 | 2.838 | 0.0118 | 5 | 8.0% |

**Key observations**

- EfficientNet-B0 is best in Accuracy, Recall, F1, ROC-AUC, and EER. Its EER of
  1.255% is clearly lower than the other models.
- ResNet18 has higher Precision than EfficientNet-B0 (0.9980 vs 0.9911), but lower
  Recall and a higher EER.
- Simple CNN has the highest Precision (0.9994) and Specificity (0.9997), but misses
  more fake clips; its EER is 2.393%.
- Both hybrids have EER around 2.83% and rank last, consistent with the earlier
  finding that the temporal models did not add value.
- EER thresholds are much lower than 0.5 (0.001-0.149), showing that the models are
  highly confident. The EER operating point is therefore different from the fixed
  0.5-threshold metrics.
- In anti-fraud use, lowering the threshold can reduce missed fakes at the cost of
  more false alarms; EER provides a balanced reference point where FPR = FNR.
- Per-speaker breakdown: Adam Driver is the hardest validation speaker for every
  model. Apart from him (and Kamala Harris, who has only 4 fake clips), every
  validation speaker has fake recall of at least 99.7% for all five models.

**Figures**

- `outputs/figures/08_metric_comparison.png`: metric heatmap plus EER bar chart.
- `outputs/figures/08_roc_comparison.png`: full ROC curves and a zoom near the
  operating region.
- `outputs/figures/08_eer_curves.png`: FPR vs FNR curves (DET-style, linear axes)
  with the FPR = FNR diagonal and EER points.
- `outputs/figures/08_confusion_matrices.png`: confusion matrices for all five
  models.

**Outputs**

- `outputs/logs/08_model_comparison.csv`
- `outputs/logs/08_model_comparison.json`
- `outputs/logs/08_model_comparison.md`
- `outputs/logs/08_per_speaker_breakdown.csv`: per-speaker FN/FP, fake recall, and
  mean fake probability for every model.

**Caveats**

- All metrics come from the same validation split used for model selection; this is
  not an independent test set and does not establish cross-dataset generalization.
- There is only one speaker-disjoint split with 14 validation speakers, so small
  differences between models (for example, ResNet18 vs Simple CNN) may not be stable.
- EER and fixed-threshold metrics answer different questions: EER measures ranking
  ability with balanced error rates, while 0.5-threshold metrics depend on probability
  calibration.
- Although EfficientNet-B0 wins overall, it still has 73 false negatives and 31 false
  positives, and only 47.4% fake recall on Adam Driver. The aggregate ranking does not
  remove the hard-speaker problem.

**Takeaway**

> On the shared validation set, EfficientNet-B0 ranks first with 0.9852 F1, 0.9991
> ROC-AUC, and 1.255% EER. ResNet18 and Simple CNN follow with EER 2.18% and 2.39%,
> while the two CNN+RNN hybrids rank last at about 2.83%. EER and AUC largely agree on
> the ordering, but EER directly answers "how high is the error when false positives
> and false negatives are balanced?" These are validation results and the validation
> set was used for model selection; a truly independent test set or cross-dataset
> evaluation is still needed before claiming generalization.

---

## Step 9: Error analysis and limitations

**Goal:** understand *where* and *why* the five models fail, check whether the
shortcuts found in Steps 2-3 still drive errors, and list the limitations of the
project. No model is retrained; the analysis uses the saved validation predictions.

**Method**

- Error overlap: for every validation clip, count how many of the five models
  misclassify it (threshold 0.5).
- Error by speaker: which validation speakers produce the false negatives (FN) and
  false positives (FP).
- Error by possible shortcut: error rate grouped by wrap-padding amount, original
  duration, and original loudness, with and without Adam Driver. Here "original
  loudness" is the RMS of the 4-second cropped/padded clip *before* loudness
  normalization (`rms_before_dbfs` from Step 3), not the full-clip RMS used in Step 2.
- Hard-speaker profile: compare Adam Driver's clips with the training clips on
  original loudness, duration, and average spectral shape (per-band mean log-Mel
  with each clip's mean removed, as in Step 2). The spectral profile uses 1,500
  random clips per class for the training set and for the other validation
  speakers, and all of Adam Driver's clips. "High band" means Mel bands 96-127
  (about 3.6-8 kHz).
- Threshold trade-off: Recall, FPR, and Adam Driver recall when the threshold is set
  for FPR <= 0.1%, 1%, or 5% instead of 0.5.
- Hard examples: the 10 most confident FN and FP of EfficientNet-B0, with file paths
  for manual listening.

**Key numbers**

| Model | FN (missed fakes) | of which Adam Driver | FP (false alarms) | Main FP speaker |
|---|---:|---:|---:|---|
| Simple CNN | 139 | 134 (96.4%) | 2 | 2Pac (2) |
| ResNet18 | 115 | 114 (99.1%) | 7 | Ayn Rand, Barack Obama (3 each) |
| EfficientNet-B0 | 73 | 72 (98.6%) | 31 | Dwight Eisenhower (21) |
| CNN+BiGRU | 129 | 126 (97.7%) | 16 | 2Pac (11) |
| CNN+BiLSTM | 121 | 120 (99.2%) | 18 | 2Pac (12) |

- **Missed fakes are almost all one speaker.** Across all models, 96-99% of FN come
  from Adam Driver. 69 clips are missed by all five models, and all 69 are Adam
  Driver fakes. Every one of his 137 fake clips is missed by at least one model.
- **False alarms are model-specific.** No real clip is misclassified by three or more
  models. EfficientNet-B0's FP are concentrated on Dwight Eisenhower (21 of 31), a
  quiet, old recording (median original loudness -30.9 dBFS); the hybrids' FP are
  mostly 2Pac. EfficientNet-B0 also has the most errors that no other model makes
  (29), while the other models have only 5-10 unique errors each.
- **Padding and duration do not drive errors.** Without Adam Driver, the average
  miss rate on fakes is 0-0.2% in every padding and duration bin. The only weak
  effect is on very short clips (3-4 s of padding, i.e. clips under 1 s): the miss
  rate there is 1.1% (6 errors over 5 models x 107 clips), and EfficientNet-B0 FPR is
  1.9% (4 of 208) versus at most 0.7% in the other padding bins. The Step 3 worry about the
  padding shortcut is not confirmed.
- **Loudness appears to matter only through Adam Driver.** With all speakers, missed
  fakes rise sharply for originally loud fakes (above -20 dBFS: 24-51% missed), but
  after removing Adam Driver the miss rate in the two loudest bins drops to 0-1.4%
  per model (at most 2 missed clips in any bin; 0.1-0.3% averaged over the five models).

**Why is Adam Driver hard?**

| Group | Clips | Median original loudness (dBFS) | Median duration (s) | High-band (3.6-8 kHz) relative energy (dB) |
|---|---:|---:|---:|---:|
| Train Real | 13,954 | -25.8 | 2.87 | -11.9 (1,500-clip sample) |
| Train Fake | 8,279 | -28.7 | 4.17 | -13.5 (1,500-clip sample) |
| Other validation Fake | 3,400 | -28.3 | 3.76 | -14.6 (1,500-clip sample) |
| **Adam Driver Fake** | 137 | **-19.7** | 3.83 | **-10.1** |
| Adam Driver Real | 80 | -18.8 | 5.87 | -15.9 |

- His fake clips were recorded/produced about 9 dB louder than typical training
  fakes; only 1.4% of training fakes are as loud (mostly Nelson Mandela, Martin
  Luther King, and Donald Trump). Loudness itself is normalized before feature
  extraction, so this points to a different generation or recording pipeline rather
  than a direct loudness cue.
- Spectrally, his fakes look more like **real** training speech than like training
  fakes: profile distance (RMSE) 2.0 dB to Train Real vs 3.4 dB to Train Fake. Other
  validation fakes show the opposite pattern (2.7 dB to Train Fake, 3.1 dB to Train Real).
- His fakes have the **most** high-frequency energy of any group (-10.1 dB), even
  more than real speech. Step 2 found that real speech has relatively more upper-band
  energy than fakes, so a model that learned "more high-frequency energy = real"
  would call these fakes real. This is a likely explanation, not a proven cause.
- The models are confidently wrong: for every model except EfficientNet-B0, 59-82%
  of his fake clips get a fake probability below 0.01 (EfficientNet-B0: 12%).

**Threshold trade-off (EfficientNet-B0)**

| Operating point | Threshold | Recall | FPR | Missed fakes | False alarms | Adam Driver recall |
|---|---:|---:|---:|---:|---:|---:|
| FPR <= 0.1% | 0.915 | 97.0% | 0.10% | 105 | 6 | 25.5% |
| Threshold 0.5 | 0.500 | 97.9% | 0.52% | 73 | 31 | 47.4% |
| FPR <= 1% | 0.244 | 98.5% | 1.00% | 52 | 60 | 62.8% |
| FPR <= 5% | 0.004 | 99.7% | 4.83% | 12 | 290 | 91.2% |

- Lowering the threshold helps EfficientNet-B0 recover Adam Driver's fakes, but at a
  growing cost: moving from 0.5 to the FPR <= 1% point catches 21 more fakes for 29
  more false alarms (about 1.4 per fake); moving on to FPR <= 5% catches 40 more
  fakes for 230 more false alarms (about 6 per fake). For the other models, the threshold
  cannot fix the problem: at FPR <= 1%, Adam Driver recall is only 17-31%.

**Figures**

- `outputs/figures/09_error_overlap.png`: (a) validation clips grouped by how many of
  the five models misclassify them (the x-axis is a count, not a model ID); (b) each
  model's errors split into errors only that model makes, errors shared with 1-3
  other models, and errors shared by all five.
- `outputs/figures/09_error_by_confounder.png`: FNR and FPR by padding, original
  duration, and original loudness for all models.
- `outputs/figures/09_adam_driver_profile.png`: (a) original loudness of his fakes vs
  training clips; (b) average spectral shape relative to training fakes; (c)
  cumulative distribution of the fake probability for his fake clips.
- `outputs/figures/09_threshold_tradeoff.png`: Recall, FPR, and Adam Driver recall as
  the threshold changes.

**Outputs**

- `outputs/logs/09_error_summary.json`: error counts, overlap, and spectral statistics.
- `outputs/logs/09_error_overlap.csv`
- `outputs/logs/09_error_overlap_by_model.csv`: each model's unique vs shared errors.
- `outputs/logs/09_error_by_bin.csv`: error rate per bin, class, and model (with and
  without Adam Driver).
- `outputs/logs/09_speaker_profile.csv`
- `outputs/logs/09_threshold_analysis.csv`
- `outputs/logs/09_hard_examples.csv`: most confident mistakes with file paths.

**Limitations**

- **One split, few validation speakers.** Results come from a single speaker-disjoint
  split with 14 validation speakers. One speaker (Adam Driver, 137 fake clips)
  accounts for almost all missed fakes, so the ranking of models depends heavily on
  which speakers happen to be in validation. Speaker-level cross-validation would
  give a more reliable estimate.
- **No independent test set.** The validation set was used for early stopping and
  model selection, so all reported numbers are slightly optimistic.
- **One dataset.** All models were trained and tested on In-the-Wild only. Nothing
  here shows that they work on other datasets, newer voice-cloning tools, phone-call
  audio, or compressed audio.
- **Unknown generation methods.** The dataset does not say which tool produced each
  fake, so errors cannot be analyzed per generator. Adam Driver's fakes probably come
  from a different pipeline, but this cannot be confirmed.
- **Fixed 4-second input.** Long clips are center-cropped (content outside the crop is
  ignored) and short clips are repeated. For EfficientNet-B0 and ResNet18 this
  slightly raises false alarms on clips shorter than 1 second (1.9% and 1.4%); the
  other three models have none there.
- **Overconfident probabilities.** EER thresholds are far below 0.5 and many mistakes
  have near-0 or near-1 probabilities, so the raw probability is not a reliable
  confidence score without calibration.
- **No shortcut ablation.** The planned control run without loudness normalization
  (`mel_raw`) was not done, so we cannot measure directly how much a model would
  rely on loudness if it were not removed.
- **Correlational explanation.** The high-frequency explanation for Adam Driver is based on
  average spectra, not on listening tests or model-attribution methods (for example,
  saliency maps).
- **Single training run per model.** Each model was trained once with seed 42, so
  run-to-run variance is unknown; small differences between models may not be
  meaningful.

**Takeaway**

> Error analysis shows that the models fail in a very concentrated way: 96-99% of
> missed fakes come from a single validation speaker, Adam Driver, and 69 of his
> fakes fool all five models. His fake clips were produced much louder than typical
> training fakes and carry more high-frequency energy than even real speech, so they
> look "real" to models that learned high-frequency energy as a sign of authenticity.
> Padding and duration, the shortcuts we worried about after preprocessing, do not
> explain the errors. The lesson is that a 99% AUC on one dataset can still hide a
> complete failure on a new voice or generation pipeline; per-speaker evaluation and
> cross-dataset testing are essential before trusting a deepfake detector.

---

## Repository notes

- `src/00_environment_check.py`: environment check (English output).
- `src/01_organize_dataset.py`: organize the flat audio files and build `dataset_index.csv`.
- `src/02_eda.py`: run the full EDA and regenerate all Step 2 figures.
- `src/03_preprocess.py`: build the log-Mel feature matrix and Step 3 figures
  (`--figures-only` regenerates the figures and summary from existing features).
- `src/04_split_dataloader.py`: create the speaker-disjoint split, compute training
  statistics, and smoke-test the DataLoaders.
- `src/05_train_cnn.py`: train and evaluate the simple CNN baseline; use
  `--smoke-test` for a single forward pass.
- `src/06_train_transfer.py`: fine-tune pretrained ResNet18 and EfficientNet-B0;
  `--model both` trains both sequentially.
- `src/07_train_hybrid.py`: train CNN+BiGRU and CNN+BiLSTM hybrids; `--model both`
  trains both sequentially.
- `src/08_evaluate_models.py`: compare every model on the shared validation split,
  compute EER and per-speaker breakdowns, and generate the Step 8 tables and figures.
- `src/09_error_analysis.py`: error overlap, error by padding/duration/loudness, the
  Adam Driver profile, threshold trade-off, and hard examples (Step 9).
