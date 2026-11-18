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
To build our final deepfake voice detector, EfficientNet-B0 was selected as the optimal model after comparing custom baselines and traditional transfer-learning architectures.

1. Architectural Design & Rationale
   EfficientNet-B0 introduces a paradigm shift from traditional convolutional networks by utilizing compound scaling, uniformly balancing three critical dimensions: depth (layer count), width (channel capacity),    and resolution ($224 \times 224$ input size).

   - MBConv Blocks: The network relies on Mobile Inverted Bottleneck Convolutions (MBConv) incorporating depthwise separable convolutions and squeeze-and-excitation (SE) modules. This allows it to capture fine-       grained, complex time-frequency spectrogram patterns while maintaining a lightweight footprint of $4.01\text{ million}$ trainable parameters (significantly more parameter-efficient than heavier            architectures like ResNet18's $11.18\text{ million}$).
   - Swish Activation & NAS: It employs the self-gated Swish activation function in place of standard ReLU and was optimized using an automated Neural Architecture Search (NAS) to maximize accuracy with minimal       computational overhead.

2. Input Adaptation & Training Setup
   Because computer vision models expect multi-channel 2D image tensors, log-Mel spectrograms ($128 \times 126$ matrices) underwent specialized preprocessing:
   - Normalization & Resizing: Saved dB values ($-80\text{ dB}$ to $0\text{ dB}$) were scaled to $[0, 1]$ and bilinearly resized to $224 \times 224$ pixels.
   - Channel Replication: The single-channel spectrogram was replicated across three channels to match ImageNet pre-trained expectations and normalized using standard ImageNet mean and standard deviation tensors.
   - Optimization: Training utilized differential learning rates ($\text{backbone } 1\text{e-}4$, $\text{head } 1\text{e-}3$), the AdamW optimizer (weight decay $1\text{e-}4$), class-weighted CrossEntropyLoss, a ReduceLROnPlateau scheduler, and training-time SpecAugment (time and frequency masking). It completed training in $33.1\text{ minutes}$ (best epoch $5$).
  
3. Performance & Validation Validation
On our speaker-disjoint validation split ($9,546$ clips across $14$ unseen speakers), EfficientNet-B0 achieved top-tier evaluation scores:
- Accuracy: 0.9891
- F1-Score: 0.9852
- ROC-AUC: 0.9991
- Equal Error Rate (EER): 1.255\%

### Code / features explanations

## Evaluation & Analysis
Graphs, tables, any statistics (probably the output)

## Related Work
the tools, libraries, blog, any documentations used for this project
Dominik Müller et al., 2022, Voice Spoofing in the Wild: A Large-Scale Dataset and Evaluation of Countermeasures, https://huggingface.co/papers/2203.16263
Tapendu Karmakar, 2024, Introduction to EfficientNet. Kaggle.Com; Kaggle. https://www.kaggle.com/code/iamtapendu/introduction-to-efficientnet

## Conclusion : Discussion
