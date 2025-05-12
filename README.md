# 🗣️ Vietlish Accent Conversion: Towards Hight Quality English Accent Conversion for Vietnamese people

**Vietlish Accent Conversion** is an advanced framework that merges [FreeVC](https://github.com/OlaWod/FreeVC) and [VITS](https://github.com/jaywalnut310/vits), augmented with:
- ✅ **Text encoder** for optional text-conditioned voice conversion
- 🔁 **Monotonic Alignment Search (MAS)** for robust alignment between audio and text
- 🎓 **Knowledge distillation** from a native TTS model
- 🎤 **One-shot speaker transfer**
- 🧠 **Pronunciation correction** and accent conversion for non-native speech

---

## ✨ Highlights

- Supports **text-free** and **text-conditioned** inference
- Uses **WavLM** content encoders
- Improves **non-native pronunciation** using synthetic native ground-truth data
- Aligns text and audio using **MAS**, avoids unstable attention mechanisms
- Trained with **shared decoder and distillation loss** between audio/text pathways

---


## 🎯 Motivation

While previous VC methods (like FreeVC) successfully convert voices across speakers, they often struggle with **accented or mispronounced speech**.Vietlish Accent Conversion solves this by:

- Training on synthetic **native-style ground-truth** generated from a native TTS system
- Using **MAS** to align text and audio representations
- Distilling knowledge from native representations into the audio-based encoder
- Maintaining **speaker identity, prosody, and duration**

---

## 🧱 Architecture Overview

Vietlish Accent Conversionis built on a **VITS-style** conditional variational autoencoder with shared components between TTS and VC paths:

### 🔨 Components:

| Component         | Description |
|------------------|-------------|
| **Posterior Encoder** | Encodes linear spectrogram `x` to latent `z` |
| **Prior Encoders**    | - `θ_audio`: audio path using WavLM/Wav2Vec2.0 + bottleneck extractor<br> - `θ_text`: text path with MAS alignment |
| **Normalizing Flow** | Transforms latent prior `z` → posterior space |
| **HiFi-GAN Decoder** | Decodes latent `z` + speaker/F0 into waveform |
| **F0 Encoder**        | Encodes pitch information |
| **Speaker Encoder**  | Produces speaker embedding |
| **MAS**              | Aligns text and speech latent representations |
| **Distillation Loss**| KL divergence between audio/text priors |

---

### 🧪 Training Phases

#### **1. Pretraining (Dual Path Training)**

- Train TTS (text-to-speech) model and pretrain AC (accent conversion) model **jointly**.
- Shared decoder, speaker encoder, F0 encoder, normalizing flow.
- Text path uses MAS for alignment, audio path uses WavLM or Wav2Vec2.0 + bottleneck extractor.

#### **2. Synthetic Ground Truth Generation**

- Use native TTS (text path) to generate native-style audio from non-native transcripts.
- Match speaker identity and prosody using speaker & F0 embeddings from non-native audio.

#### **3. Finetuning (Distillation and Correction)**

- Finetune the audio-based AC model using:
  - Non-native audio (input)
  - Synthetic native audio (ground-truth)
  - Distillation loss between `p(z | c_text)` and `p(z | c_audio)`

---
