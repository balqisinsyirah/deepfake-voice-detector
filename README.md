# Deepfake Voice Detector
Hanyang University, AI and Application, ITE351 (Fall 2026)

## Group Members
1. Zeng Junxiang, Information Systems, Hanyang University, goblin888@hanyang.ac.kr
2. Pa Pa Khine, Information Systems, Hanyang University, khinpapa302@gmail.com
3. Balqis Insyirah, Information Systems, Hanyang University, balqis.ink@gmail.com
4. Shaqweeza, Information Systems, Hanyang University, shqweeza20@hanyang.ac.kr

## Video Presentation
[YouTube Video Presentation](https://youtu.be/qHKQiQRuZQI)

## Table of Contents
- [I. Introduction](#i-introduction)
- [II. Datasets](#ii-datasets)
- [III. Methodology](#iii-methodology)
- [IV. Evaluation & Analysis](#iv-evaluation--analysis)
- [V. Related Work](#v-related-work)
- [VI. Conclusion: Discussion](#vi-conclusion-discussion)

## I. Introduction
The rapid advancement of generative AI has fueled a surge in sophisticated voice phishing scams, where synthetic voice clones impersonate individuals using deepfake technology to deceive victims into financial fraud. Recent studies by researchers from Harvard and Meta reveal that approximately 1 in 6 adults could comply with emotional and relationship-based AI voice-cloning scams, such as a family emergency, which is more persuasive than the typical call center impersonation (Heiding et al., 2026). Researchers warn that as this technology spreads, scam calls will become fully automated and the cost of mass dialing will plummet, making scalable defense mechanisms urgent (Heiding et al., 2026).

Our objective is to develop a baseline deepfake voice detector by training and comparing the performance of various models to combat this increasing problem. To accomplish this, we use the [In-the-Wild dataset](https://huggingface.co/datasets/mueller91/In-The-Wild) introduced by Müller et al. (2022), which contains 31,779 real and fake audio recordings from public figures and is released under CC BY-SA 4.0. We split the dataset by speaker into two parts: 70% for training and 30% for the validation set, process raw audio files via librosa to extract log-Mel spectrogram features, and train image-based and hybrid neural network architectures on a local NVIDIA RTX 3060 GPU.

Once trained, the models are thoroughly evaluated to assess their effectiveness, including an error analysis of where and why they fail. Ultimately, this project explores practical data-driven pipelines to safeguard individuals against automated audio-based security threats.

## II. Datasets
We use the **In-the-Wild** dataset by Müller et al. (2022), from the paper *"Does Audio Deepfake Detection Generalize?"* (Interspeech 2022). It is free to use under the CC BY-SA 4.0 license.

- **Download:** [Hugging Face – mueller91/In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) (also mirrored on [Kaggle](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake))
- **What's inside:** real and fake speech clips of **54 public figures** (politicians, actors, celebrities), collected from social media and video sites. "In the wild" means the audio is messy and realistic, not recorded in a clean studio.

<div align="center">

| | Real (bona-fide) | Fake (spoof) | Total |
|---|---:|---:|---:|
| Clips | 19,963 (62.8%) | 11,816 (37.2%) | **31,779** |
| Hours of audio | 20.8 h | 17.1 h | 37.9 h |
| Median clip length | 3.0 s | 4.1 s | |
| Median loudness | −26.1 dBFS | −28.5 dBFS | |

</div>
<p align="center">Figure 1. Overview of the In-the-Wild dataset statistics by class.</p>
<p align="center">*All clips are 16 kHz mono, with no clipping.</p>

**Two things we noticed while exploring the data:**

1. **The classes are not balanced.** There are about 1.7 real clips for every fake clip. A lazy model that always says "real" would already get 63% accuracy, so we cannot rely on accuracy alone. We also report precision, recall, F1, ROC-AUC and EER.
2. **The dataset has "shortcuts".** Fake clips are, on average, **longer** (by about 1 second for the same speaker) and **quieter** (by about 2.4 dB). Loudness alone could already separate real from fake fairly well for many speakers (median per-speaker AUC 0.744). A model could "cheat" by learning *"quiet and long = fake"* instead of learning what an AI voice actually sounds like. We fix this in preprocessing (Section III.2).

## III. Methodology

### Feature Extraction: log-Mel Spectrograms
Image-based neural networks are very good at finding patterns in pictures, so we convert every audio clip into a picture called a log-Mel spectrogram. The horizontal axis shows time, the vertical axis shows pitch (frequency), and the brightness shows how much energy the voice has at that pitch. The Mel scale spaces the pitches the way human ears hear them, and the log (decibel) scale does the same for loudness. Synthetic voices often leave small traces in this picture, especially in the high frequencies, that a model can learn to spot.

Using librosa, each clip is analysed with an FFT window of 2,048 samples and a hop length of 512, mapped onto 128 Mel bands between 20 Hz and 8 kHz, and converted to decibels between −80 and 0 dB. Every clip becomes a 128 × 126 image.

<p align="center"><img width="2456" height="1359" alt="03_preprocessed_examples" src="https://github.com/user-attachments/assets/90d3f0cb-5745-47cf-9223-898864b0e6a0" /></p>
<p align="center">*Figure 2. Waveform and log-Mel spectrogram of a real and a fake clip from the same speaker.*</p>

### Preprocessing
To remove the shortcuts found in Section II, every clip is made identical in length and loudness before it is converted. Each clip is fixed to exactly 4 seconds: longer clips keep their middle 4 seconds, and shorter clips are repeated (looped) until they reach 4 seconds, instead of being padded with silence, which could become a new clue. Every clip is also normalized to the same loudness (−23 dBFS). After this step, loudness alone can no longer separate real from fake (its AUC dropped from 0.649 to 0.505, which is close to random guessing).

### Data Split
The data is split 70% for training and 30% for validation by speaker, meaning every clip of one person goes into only one of the two sets. The models are therefore tested on 14 voices they have never heard during training (40 speakers for training, 14 for validation), which is closer to real life, where a scammer can clone anyone's voice. Both sets keep the same ratio of fake clips (about 37%).

### Models
We trained and compared five models.

**Simple CNN (baseline).** A small convolutional neural network built from scratch, with four convolution layers (32, 64, 128 and 256 channels) and about 0.39 million parameters. It shows how far a basic approach can go.

**ResNet18 and EfficientNet-B0 (transfer learning).** Two well-known image models that were already trained on millions of photos from ImageNet. We fine-tune them on our spectrograms to test whether knowledge learned from photos also helps with audio. Because these models expect 224 × 224 colour images, each spectrogram is resized and copied into three channels, and the last layer is replaced with a new two-class (real/fake) layer. EfficientNet-B0 uses a compound scaling method that balances the depth, width and input size of the network, giving strong performance with only 4 million parameters, compared with 11 million for ResNet18.

**CNN + BiGRU and CNN + BiLSTM (hybrid).** A small CNN first finds local patterns in the spectrogram, then a bidirectional recurrent network (GRU or LSTM) reads these patterns in time order, both forwards and backwards. An attention layer then decides which moments in the clip matter most. These models test whether looking at how the voice changes over time helps to catch fakes.

### Training Setup
All five models were trained with the same rules so the comparison is fair. We used cross-entropy loss with class weights, so that missing a fake clip is punished more than missing a real one, and the AdamW optimizer. The pretrained models used a smaller learning rate (1e-4) for their existing layers and a larger one (1e-3) for the new final layer, so they keep what they already know. During training we applied SpecAugment, which randomly hides small time and frequency strips of the spectrogram so that the model cannot simply memorize the training images. After every epoch the model was checked on the validation set, the best version was kept, and training stopped early once it stopped improving. Training took between 3 and 33 minutes per model.

The main libraries used are PyTorch and torchvision (models), librosa and soundfile (audio), scikit-learn and SciPy (metrics and statistics), NumPy and pandas (data handling), and Matplotlib and seaborn (figures).

### Brief Code Explanations
The main files of our project and their tasks are listed below. Click a file name to see the full code.

#### [00_environment_check.py](src/00_environment_check.py)
Checks that the computer is ready before anything else runs: the Python version, the required libraries, the GPU, the free disk space, and whether the dataset files are in place.

#### [01_organize_dataset.py](src/01_organize_dataset.py)
Reads the dataset's metadata file, renames the original labels (bona-fide → real, spoof → fake), checks that no file is missing, and moves the audio into `real/` and `fake/` folders. It also creates an index table listing each clip's speaker, label and file path.

#### [02_eda.py](src/02_eda.py)
Explores the data by measuring every clip's length, sample rate and loudness, and uses statistical tests to check whether real and fake clips differ. This is where we discovered the length and loudness shortcuts. It also draws comparison plots of waveforms and spectrograms.

#### [03_preprocess.py](src/03_preprocess.py)
Makes every clip 4 seconds long and the same loudness, then converts it into a 128 × 126 log-Mel spectrogram. All 31,779 spectrograms are saved into a single file so the models can load them quickly.

#### [04_split_dataloader.py](src/04_split_dataloader.py)
Divides the 54 speakers into a 70% training set and a 30% validation set while keeping the same ratio of fake clips in both. It computes the normalization values from the training set only, so no information from the validation set leaks into training, and prepares the data loaders that feed the models in batches of 32.

#### [05_train_cnn.py](src/05_train_cnn.py)
Trains the simple CNN baseline. The best result was reached at epoch 12, and training took about 5 minutes.

#### [06_train_transfer.py](src/06_train_transfer.py)
Fine-tunes the pretrained ResNet18 and EfficientNet-B0 models. ResNet18 took about 21 minutes and EfficientNet-B0 about 33 minutes.

#### [07_train_hybrid.py](src/07_train_hybrid.py)
Trains the two hybrid models, CNN + BiGRU and CNN + BiLSTM, which took about 3 and 6.5 minutes respectively.

#### [08_evaluate_models.py](src/08_evaluate_models.py)
Compares all five models on the same validation clips using accuracy, precision, recall, F1, ROC-AUC and EER, and draws the comparison charts. It also breaks the results down per speaker, which revealed the Adam Driver problem described in Section IV.

#### [09_error_analysis.py](src/09_error_analysis.py)
Studies the mistakes without retraining anything: whether the models fail on the same clips, which speakers cause the errors, whether clip length or loudness still affects the errors, and how changing the decision threshold trades missed fakes against false alarms.

## IV. Evaluation & Analysis
All models were tested on the same 9,546 validation clips from 14 unseen speakers. Precision tells us how many of the clips flagged as fake were really fake, recall tells us how many of the fake clips were caught, and F1 balances the two. ROC-AUC measures how well the model ranks fake clips above real ones (1.0 is perfect). EER (Equal Error Rate) is the error rate at the point where false alarms and missed fakes are equal, so lower is better.

<div align="center">

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | **98.91%** | 99.11% | **97.94%** | **0.9852** | **0.9991** | **1.26%** |
| ResNet18 | 98.72% | 99.80% | 96.75% | 0.9825 | 0.9975 | 2.18% |
| Simple CNN | 98.52% | **99.94%** | 96.07% | 0.9797 | 0.9957 | 2.39% |
| CNN + BiLSTM | 98.54% | 99.48% | 96.58% | 0.9801 | 0.9916 | 2.83% |
| CNN + BiGRU | 98.48% | 99.53% | 96.35% | 0.9792 | 0.9889 | 2.84% |

</div>

<p align="center">*Figure 3. Comparison of all five models.*</p>

<p align="center"><img width="695" height="550" alt="08_eer_curves" src="https://github.com/user-attachments/assets/870a82c8-9630-4873-b611-fcfe93cc837f" /></p>

<p align="center">*Figure 4. Comparison of False Positive Rate (FPR) versus False Negative Rate (FNR) and Equal Error Rate (EER) across all five models.*</p>

EfficientNet-B0 is the best model overall, with the highest recall, F1 and ROC-AUC and the lowest EER. However, the simple CNN is surprisingly close, which shows that log-Mel spectrograms combined with a basic CNN are already a strong baseline. The hybrid models ranked last, so analysing the sound in time order did not help. A likely reason is that each clip only has 31 time steps, and the traces left by synthetic voices seem to be local rather than spread over time.

Although the overall scores look almost perfect, breaking them down by speaker revealed a major weakness. Between 96% and 99% of the missed fakes in every model came from a single speaker, Adam Driver, and 69 of his fake clips fooled all five models. Even the best model caught only 47.4% of his fakes, while for every other validation speaker the models caught almost all fakes. His fake clips contain more high-frequency energy than even real speech, and since the models had learned that strong high frequencies usually mean a real voice, they confidently labelled his fakes as real. His fakes were also originally much louder than the training fakes, which suggests they were made with a different voice-cloning tool. After removing Adam Driver, the errors no longer depended on clip length or loudness, which confirms that our preprocessing successfully removed those shortcuts.

<p align="center"><img width="3999" height="1100" alt="09_adam_driver_profile" src="https://github.com/user-attachments/assets/a605dccf-d927-4a9d-aa32-d9e3e5b89baf" /></p>
<p align="center">*Figure 5. Adam Driver's fake clips compared with the training data.*</p>

Finally, by default a clip is labelled fake when the model's fake probability is above 0.5. Lowering this threshold catches more fakes but also produces more false alarms. For EfficientNet-B0, lowering it to 0.244 raised overall recall from 97.9% to 98.5% and Adam Driver recall from 47.4% to 62.8%, at the cost of 29 additional false alarms. In a scam-call setting, where missing a fake is worse than a false alarm, a lower threshold may be the better choice.

<p align="center"><img width="3998" height="1100" alt="09_threshold_tradeoff" src="https://github.com/user-attachments/assets/38d63ed3-2470-4b91-8b32-a41661d0be80" /></p>
<p align="center">*Figure 6. Effect of decision threshold trade-off on overall fake recall, FPR and Adam Driver fake recall across all five models.*</p>

## V. Related Work
The tools, libraries, papers and documentation used for this project:
- Heiding, F., Mayrink Verdun, C., Lermen, S., Kao, A., Albiero, V., Deason, L., Veliche, I.-E., & Lehane, C. (2026). Evaluating AI Models' Capability to Automate Voice Phishing Attacks. *Expert Systems with Applications*, 332. https://doi.org/10.1016/j.eswa.2026.133620
- Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A., & Böttinger, K. (2022). Does Audio Deepfake Detection Generalize? *Interspeech 2022*. https://arxiv.org/abs/2203.16263
- Pham, L. et al. (2024). Deepfake audio detection using spectrogram-based feature and ensemble of deep learning models. arXiv:2407.01777. https://arxiv.org/abs/2407.01777
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*. https://arxiv.org/abs/1905.11946
- Deng, J. et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR 2009*. https://doi.org/10.1109/CVPR.2009.5206848
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP 2014*. https://arxiv.org/abs/1406.1078
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*. https://arxiv.org/abs/1409.0473
- Park, D. S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech 2019*. https://arxiv.org/abs/1904.08779
- Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*. https://arxiv.org/abs/1711.05101
- McFee, B. et al. (2015). librosa: Audio and Music Signal Analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–24. https://doi.org/10.25080/Majora-7b98e3ed-003
- Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*. https://pytorch.org
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. https://scikit-learn.org
- Karmakar, T. (2024). Introduction to EfficientNet. Kaggle. https://www.kaggle.com/code/iamtapendu/introduction-to-efficientnet

## VI. Conclusion: Discussion
In this project we built a complete pipeline that converts speech into log-Mel spectrograms, removes the length and loudness shortcuts present in the dataset, and compares five neural networks on speakers they have never heard. EfficientNet-B0 performed best, with an F1 score of 0.985, a ROC-AUC of 0.999 and an EER of 1.26%, although a simple CNN came surprisingly close. Adding recurrent layers to model time did not improve the results, showing that a more complex model is not automatically a better one.

The most important lesson came from the error analysis. A score of 99% on average can still hide a complete failure: almost all missed fakes came from a single speaker whose fake clips were probably produced with a different voice-cloning tool. This is exactly the situation a real detector would face, since scammers constantly use new tools.

Our work has several limitations. We used only one train/validation split with 14 validation speakers, so the results depend on which speakers ended up in validation. The validation set was also used to choose the best epoch, so the scores are slightly optimistic, and each model was trained only once. Finally, we only tested on one dataset. Therefore, we couldn't confirm whether the same models would work the same on other datasets. In future work, we would like to test on other datasets and real phone-call audio, use cross-validation across speakers, include training data from more voice-cloning tools, and use explanation methods to see which parts of the spectrogram the models actually focus on.
