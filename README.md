# Deepfake Voice Detector
Hanyang University, AI and Application, ITE351 (Fall 2026)

 ## Group Members
 1. Zeng Junxiang, Information Systems, goblin888@hanyang.ac.kr
 2. Pa Pa Khine, Information Systems, khinpapa302@gmail.com
 3. Balqis Insyirah, Information Systems, balqis.ink@gmail.com
 4. Shaqweeza, Information Systems, shqweeza20@hanyang.ac.kr

## Introduction
The rapid advancement of generative AI has fueled a surge in sophisticated voice phishing scams, where synthetic voice clones impersonate individuals using deepfake technology to deceive victims into financial fraud. Recent studies by researchers from Harvard and Meta reveal that approximately 1 in 6 adults could comply with emotional and relationship-based AI voice-cloning scams, such as a family emergency, which is more persuasive than the typical call center impersonation. Researchers warn that as this technology spreads, scam calls will become fully automated and the cost of mass dialing will plummet, making scalable defense mechanisms urgent.

Our objective is to develop a baseline deepfake voice detector by training and comparing the performance of various models to combat this increasing problem. To accomplish this, we utilized this Kaggle dataset, which contains approximately 31,800 audio recordings. We will split the dataset into two parts: 70% for training and 30% for the validation set, processing raw audio files via librosa to extract Mel-spectrogram features, and training image-based and hybrid neural network architectures using AWS SageMaker Studio Lab.

Once trained, the models will be thoroughly evaluated to assess their effectiveness, with potential scope expansions depending on the results. Ultimately, this project explores practical data-driven pipelines to safeguard individuals against automated audio-based security threats.

## Datasets
In this project, we use a Kaggle dataset titled 'In-the-Wild' created by Abdalla Mohamed and Mohammed Abdeldayem. The dataset includes audio deepfakes and corresponding bona-fide (real) audio recordings for a set of politicians and other public figures, collected from publicly available sources such as social networks and video streaming platforms. The dataset contains approximately 31,800 audio recordings for n = 58 politicians and celebrities, divided into three main subsets with a total duration of 38 hours (20.8 hours of bona-fide audio and 17.2 hours of spoofed audio). On average, there are 23 minutes of bona-fide and 18 minutes of spoofed audio per speaker.

The dataset was processed to optimize outcomes for analysis of deepfake detection and voice anti-spoofing machine-learning models. Raw audio files were processed using librosa to extract Mel-spectrogram features, transforming temporal audio waveforms into time-frequency representations. In addition, the dataset is split into 70% for training and 30% for validation. This step standardizes the input pipeline, improving the models' ability to generalize across realistic, in-the-wild audio samples. The goal of preprocessing the dataset was to ensure consistency, eliminate noise variances, and maintain compatibility with the input requirements of our selected image-based and hybrid neural network architectures in AWS SageMaker Studio Lab.

## Methodology


## Evaluation & Analysis


## Related Work
'In-the-Wild' Dataset, by Abdalla Mohamed and Mohammed Abdeldayem : https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake/data?select=release_in_the_wild

## Conclusion : Discussion
