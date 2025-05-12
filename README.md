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
