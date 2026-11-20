# Deepfake Voice Detector
Hanyang University, AI and Application, ITE351 (Fall 2026)

 ## Group Members
 1. Zeng Junxiang, Information Systems, goblin888@hanyang.ac.kr
 2. Pa Pa Khine, Information Systems, khinpapa302@gmail.com
 3. Balqis Insyirah, Information Systems, balqis.ink@gmail.com
 4. Shaqweeza, Information Systems, shqweeza20@hanyang.ac.kr

## I. Introduction
The rapid advancement of generative AI has fueled a surge in sophisticated voice phishing scams, where synthetic voice clones impersonate individuals using deepfake technology to deceive victims into financial fraud. Recent studies by researchers from Harvard and Meta reveal that approximately 1 in 6 adults could comply with emotional and relationship-based AI voice-cloning scams, such as a family emergency, which is more persuasive than the typical call center impersonation. Researchers warn that as this technology spreads, scam calls will become fully automated and the cost of mass dialing will plummet, making scalable defense mechanisms urgent.

Our objective is to develop a baseline deepfake voice detector by training and comparing the performance of various models to combat this increasing problem. To accomplish this, we use the In-the-Wild dataset introduced by Müller et al. (2022), which contains 31,779 real and fake audio recordings from public figures and is released under CC BY-SA 4.0. We will split the dataset into two parts: 70% for training and 30% for the validation set, processing raw audio files via librosa to extract Mel-spectrogram features, and training image-based and hybrid neural network architectures using AWS SageMaker Studio Lab.

Once trained, the models will be thoroughly evaluated to assess their effectiveness, with potential scope expansions depending on the results. Ultimately, this project explores practical data-driven pipelines to safeguard individuals against automated audio-based security threats.

## II. Datasets
In this project, we use the In-the-Wild dataset introduced by Müller et al. (2022) in their paper "Voice Spoofing in the Wild: A Large-Scale Dataset and Evaluation of Countermeasures". Released under the CC BY-SA 4.0 license, the dataset includes 31,779 real and fake audio recordings for public figures (such as politicians and celebrities) collected from unconstrained public sources like social networks and video streaming platforms. On average, there are 23 minutes of bona-fide (real) and 18 minutes of spoofed audio per speaker, totaling 38 hours of audio data.

The dataset was processed to optimize outcomes for evaluating deepfake detection and voice anti-spoofing machine-learning models. Raw audio files were processed using librosa to extract Mel-spectrogram features, transforming temporal audio waveforms into time-frequency representations. In addition, the dataset is split into 70% for training and 30% for validation. This step standardizes the input pipeline, improving the models' ability to generalize across realistic, in-the-wild audio samples. The goal of preprocessing the dataset was to ensure consistency, eliminate noise variances, and maintain compatibility with the input requirements of our selected image-based and hybrid neural network architectures using AWS SageMaker Studio Lab.

## III. Methodology
### Methods used from AIML libraries

### Selected CNN Model: EfficientNet-B0 //change this to model b : compared to resnet
For this project, we selected EfficientNet-B0 as our core convolutional neural network (CNN) architecture for deepfake voice detection. Originally designed via automated Neural Architecture Search (NAS), EfficientNet-B0 is well-regarded for its strong 2D spatial feature extraction capabilities and compact parameter footprint (4.01 million trainable parameters), making it well-suited for the computational demands of analyzing audio spectrograms.

EfficientNet-B0 employs a compound scaling method that systematically and uniformly balances three critical network dimensions: depth (number of layers), width (channel capacity), and resolution (224×224 input size). Built upon Mobile Inverted Bottleneck Convolutions (MBConv), squeeze-and-excitation modules, and self-gated Swish activation functions, this architecture achieves high representational capacity with fewer parameters than conventional networks. Its ability to efficiently capture subtle time-frequency variations makes it ideal for identifying the fine-grained spectral artifacts typical of synthetic speech.

To adapt our 128×126 log-Mel spectrograms for EfficientNet-B0, the decibel features (−80 dB to 0 dB) were normalized to [0,1] range, bilinearly resized to 224×224 pixels, and replicated across three channels to match ImageNet pre-trained input requirements. We replaced the default classification head (classifier[1]) with a two-class linear output layer and fine-tuned the model using differential learning rates (backbone 1e-4, head 1e-3), the AdamW optimizer, class-weighted CrossEntropyLoss, and training-time SpecAugment regularization.

### Infrastructure

### Feature : Extraction of log-Mel spectrograms using librosa
For feature extraction, raw audio waveforms were transformed into two-dimensional log-Mel spectrograms using the librosa audio processing library. Rather than feeding raw, high-dimensional time-domain signals directly into the network, converting speech into time-frequency representations enables the convolutional architecture to analyze both spectral distribution and temporal dynamics simultaneously.

To ensure uniformity across the dataset, all audio recordings were first standardized to a mono channel at a fixed sampling rate of 16 kHz and trimmed or zero-padded to a consistent 4-second duration (64,000 samples). Using librosa, each standardized waveform was segmented into overlapping frames via a Short-Time Fourier Transform (STFT) with an FFT window size of 2,048 and a hop length of 512 samples, mapping the linear frequency spectrum onto 128 Mel-scale filter banks (up to a maximum frequency of 8,000 Hz) to mirror human auditory perception.

Finally, the resulting power spectrogram magnitudes were converted into logarithmic decibel (dB) scale (librosa.power_to_db) relative to peak power and clipped to a bounded range of −80 dB to 0 dB. This compression highlights subtle high-frequency artifacts and vocoder inconsistencies characteristic of synthetic speech, yielding a fixed-size 128×126 time-frequency matrix for every audio clip.

### Brief code explanations
In this section, the code used in this project will be described. To access full code and explanation, please click code file name below. The main files of our project and their tasks are:

#### [00_environment_check.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/00_environment_check.py)

This code serves as a pre-flight diagnostic script that verifies whether the system setup is ready before running the deepfake detection pipeline. It uses only Python’s standard library so it can execute without external dependencies to check four key components which are system and Python’s environments, required libraries including librosa, hardware acceleration, along with storage and dataset readiness to ensure the raw audio files are in place before proceeding to the next step.

#### [01_organize_dataset.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/01_organize_dataset.py)
This code structures the raw In-the-Wild audio files into binary class directories and builds a centralized metadata index for downstream processing. Using ⁠meta.csv⁠, the script maps original dataset labels (⁠bona-fide⁠ and ⁠spoof⁠) to standardized ⁠real⁠ (⁠0⁠) and ⁠fake⁠ (⁠1⁠) categories, performs a pre-move integrity scan to catch missing or conflicting files, and safely relocates the recordings into ⁠real/⁠ and ⁠fake/⁠ subdirectories when executed with the ⁠--apply⁠ flag. Finally, it generates ⁠dataset_index.csv⁠ containing each clip's speaker identity, standardized label, relative file path, and file size to support subsequent data splitting and feature extraction.

#### [02_eda.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/02_eda.py)
This code conducts an exploratory data analysis across all 31,779 clips to inspect acoustic distributions, audit for confounding metadata shortcuts, and visualize spectral patterns prior to model training. Using ⁠soundfile⁠ and ⁠scipy.stats⁠, the script extracts per-clip duration, sample rate, channel count, and RMS loudness (⁠dBFS⁠), followed by non-parametric statistical tests (Mann-Whitney U, Kolmogorov-Smirnov, and within-speaker Wilcoxon tests) and single-feature ROC-AUC checks to measure whether duration or volume alone can separate real from synthetic speech. Finally, it exports summary CSV tables alongside comparative plots of class distributions, speaker compositions, same-speaker waveforms, and log-Mel spectrograms to guide preprocessing decisions.

#### [03_preprocess.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/03_preprocess.py)
This code implements the core audio preprocessing and feature extraction pipeline, converting all raw speech waveforms into uniform two-dimensional log-Mel spectrogram representations. Using ⁠librosa⁠, the script resamples every clip to 16\text{ kHz} mono and standardizes duration to 4 seconds (64,000 samples) via center-cropping for longer recordings and wrap-padding for shorter clips to prevent artificial silence shortcuts. It then normalizes clip loudness to -23.0\text{ dBFS} RMS (with a 0.99 peak limiter) to eliminate volume bias before computing 128-band log-Mel spectrograms (⁠n_fft=2048⁠, ⁠hop_length=512⁠, ⁠fmax=8000 Hz⁠, ⁠top_db=80.0⁠), saving the resulting 128 \times 126 ⁠float32⁠ feature matrices and metadata in parallel for model training.

#### [04_split_dataloader.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/04_split_dataloader.py)

#### [05_train_cnn.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/05_train_cnn.py)
This code trains our Simple CNN baseline. It reads the log-Mel pictures, standardizes them with the training mean and standard deviation, and randomly hides small strips of each training picture (SpecAugment) so the model does not just memorizes them. The model has four convolution blocks(32, 64, 128, and 256 channels, each with BatchNorm and ReLU),global average pooling, Dropout 0.4, and a 2-class output layer, which gives about 0.39 million parameters. We train it with the AdamW optimizer and class weights, because fake clips are only about 37% of the data and a scheduler lowers the learning rate when progress slows down. After every epoch, the code checks the model on the validation set, which only has speakers the model has never heard, keeps the best epoch by ROC-AUC, and stops early if it stops improving. In our run, the best epoch was 12 and training took about 5.4 minutes. The best model, its results, and four figures are saved.

#### [06_train_transfer.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/06_train_transfer.py)

#### [07_train_hybrid.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/07_train_hybrid.py)

#### [08_evaluate_models.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/08_evaluate_models.py)

#### [09_error_analysis.py](https://github.com/balqisinsyirah/deepfake-voice-detector/blob/2b9b3417665fc2711cd0d1dfbbd0ddcde65fbe1a/src/09_error_analysis.py)


## IV. Evaluation & Analysis
Graphs, tables, any statistics (probably the output)

## Related Work
the tools, libraries, blog, any documentations used for this project
- Dominik Müller et al., 2022, Voice Spoofing in the Wild: A Large-Scale Dataset and Evaluation of Countermeasures, https://huggingface.co/papers/2203.16263
- Tapendu Karmakar, 2024, Introduction to EfficientNet. Kaggle.Com; Kaggle. https://www.kaggle.com/code/iamtapendu/introduction-to-efficientnet
- Pham L., 2024, Deepfake audio detection using spectrogram-based feature and ensemble of deep learning models (2407.01777v1). arXiv. https://arxiv.org/abs/2407.01777v1
- McFee, B. et al., 2015,  librosa: Audio and music signal analysis in Python. Proceedings of the 14th Python in Science Conference (SciPy), 18–24. https://doi.org/10.25080/Majora-7b98e3ed-003
- 
## Conclusion : Discussion
