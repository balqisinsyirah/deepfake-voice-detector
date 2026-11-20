# Deepfake Voice Detector
Hanyang University, AI and Application, ITE351 (Fall 2026)

 ## Group Members
 1. Zeng Junxiang, Information Systems, goblin888@hanyang.ac.kr
 2. Pa Pa Khine, Information Systems, khinpapa302@gmail.com
 3. Balqis Insyirah, Information Systems, balqis.ink@gmail.com
 4. Shaqweeza, Information Systems, shqweeza20@hanyang.ac.kr

## Introduction
The rapid advancement of generative AI has fueled a surge in sophisticated voice phishing scams, where synthetic voice clones impersonate individuals using deepfake technology to deceive victims into financial fraud. Recent studies by researchers from Harvard and Meta reveal that approximately 1 in 6 adults could comply with emotional and relationship-based AI voice-cloning scams, such as a family emergency, which is more persuasive than the typical call center impersonation. Researchers warn that as this technology spreads, scam calls will become fully automated and the cost of mass dialing will plummet, making scalable defense mechanisms urgent.

Our objective is to develop a baseline deepfake voice detector by training and comparing the performance of various models to combat this increasing problem. To accomplish this, we use the In-the-Wild dataset introduced by Müller et al. (2022), which contains 31,779 real and fake audio recordings from public figures and is released under CC BY-SA 4.0. We will split the dataset into two parts: 70% for training and 30% for the validation set, processing raw audio files via librosa to extract Mel-spectrogram features, and training image-based and hybrid neural network architectures using AWS SageMaker Studio Lab.

Once trained, the models will be thoroughly evaluated to assess their effectiveness, with potential scope expansions depending on the results. Ultimately, this project explores practical data-driven pipelines to safeguard individuals against automated audio-based security threats.

## Datasets
In this project, we use the In-the-Wild dataset introduced by Müller et al. (2022) in their paper "Voice Spoofing in the Wild: A Large-Scale Dataset and Evaluation of Countermeasures". Released under the CC BY-SA 4.0 license, the dataset includes 31,779 real and fake audio recordings for public figures (such as politicians and celebrities) collected from unconstrained public sources like social networks and video streaming platforms. On average, there are 23 minutes of bona-fide (real) and 18 minutes of spoofed audio per speaker, totaling 38 hours of audio data.

The dataset was processed to optimize outcomes for evaluating deepfake detection and voice anti-spoofing machine-learning models. Raw audio files were processed using librosa to extract Mel-spectrogram features, transforming temporal audio waveforms into time-frequency representations. In addition, the dataset is split into 70% for training and 30% for validation. This step standardizes the input pipeline, improving the models' ability to generalize across realistic, in-the-wild audio samples. The goal of preprocessing the dataset was to ensure consistency, eliminate noise variances, and maintain compatibility with the input requirements of our selected image-based and hybrid neural network architectures using AWS SageMaker Studio Lab.

## Methodology
### Methods used from AIML libraries

### Selected CNN Model: EfficientNet-B0
For this project, we selected EfficientNet-B0 as our core convolutional neural network (CNN) architecture for deepfake voice detection. Originally designed via automated Neural Architecture Search (NAS), EfficientNet-B0 is well-regarded for its strong 2D spatial feature extraction capabilities and compact parameter footprint (4.01 million trainable parameters), making it well-suited for the computational demands of analyzing audio spectrograms.

EfficientNet-B0 employs a compound scaling method that systematically and uniformly balances three critical network dimensions: depth (number of layers), width (channel capacity), and resolution (224×224 input size). Built upon Mobile Inverted Bottleneck Convolutions (MBConv), squeeze-and-excitation modules, and self-gated Swish activation functions, this architecture achieves high representational capacity with fewer parameters than conventional networks. Its ability to efficiently capture subtle time-frequency variations makes it ideal for identifying the fine-grained spectral artifacts typical of synthetic speech.

To adapt our 128×126 log-Mel spectrograms for EfficientNet-B0, the decibel features (−80 dB to 0 dB) were normalized to [0,1] range, bilinearly resized to 224×224 pixels, and replicated across three channels to match ImageNet pre-trained input requirements. We replaced the default classification head (classifier[1]) with a two-class linear output layer and fine-tuned the model using differential learning rates (backbone 1e-4, head 1e-3), the AdamW optimizer, class-weighted CrossEntropyLoss, and training-time SpecAugment regularization.

### Feature : Extraction of log-Mel spectrograms using librosa
For feature extraction, raw audio waveforms were transformed into two-dimensional log-Mel spectrograms using the librosa audio processing library. Rather than feeding raw, high-dimensional time-domain signals directly into the network, converting speech into time-frequency representations enables the convolutional architecture to analyze both spectral distribution and temporal dynamics simultaneously.

To ensure uniformity across the dataset, all audio recordings were first standardized to a mono channel at a fixed sampling rate of 16 kHz and trimmed or zero-padded to a consistent 4-second duration (64,000 samples). Using librosa, each standardized waveform was segmented into overlapping frames via a Short-Time Fourier Transform (STFT) with an FFT window size of 2,048 and a hop length of 512 samples, mapping the linear frequency spectrum onto 128 Mel-scale filter banks (up to a maximum frequency of 8,000 Hz) to mirror human auditory perception.

Finally, the resulting power spectrogram magnitudes were converted into logarithmic decibel (dB) scale (librosa.power_to_db) relative to peak power and clipped to a bounded range of −80 dB to 0 dB. This compression highlights subtle high-frequency artifacts and vocoder inconsistencies characteristic of synthetic speech, yielding a fixed-size 128×126 time-frequency matrix for every audio clip.

### Codes explanations
#### 00_environment_check.py

#### 01_organize_dataset.py

#### 02_eda.py

## Evaluation & Analysis
Graphs, tables, any statistics (probably the output)

## Related Work
the tools, libraries, blog, any documentations used for this project
- Dominik Müller et al., 2022, Voice Spoofing in the Wild: A Large-Scale Dataset and Evaluation of Countermeasures, https://huggingface.co/papers/2203.16263
- Tapendu Karmakar, 2024, Introduction to EfficientNet. Kaggle.Com; Kaggle. https://www.kaggle.com/code/iamtapendu/introduction-to-efficientnet
- Pham L., 2024, Deepfake audio detection using spectrogram-based feature and ensemble of deep learning models (2407.01777v1). arXiv. https://arxiv.org/abs/2407.01777v1
- McFee, B. et al., 2015,  librosa: Audio and music signal analysis in Python. Proceedings of the 14th Python in Science Conference (SciPy), 18–24. https://doi.org/10.25080/Majora-7b98e3ed-003
- 
## Conclusion : Discussion
