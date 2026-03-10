# **IndicF5: Tamil Voice Cloning**

This repository contains the fine-tuned IndicF5 model for generating high-quality Tamil voice clones, heavily trained on Swamiji's speech and native Tamil script.

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/kailasa-ngpt/IndicF5-Tamil-Finetuned)

---

## 🚀 Installation & Prerequisites

1. Ensure you have the repository cloned and your Python virtual environment activated:
   ```bash
   source venv/bin/activate
   ```
2. The necessary packages (`f5-tts`, `torch`, `torchaudio`, `huggingface_hub`, etc.) should be installed from `requirements.txt`.

---

## 🎙️ Running Local Inference

To generate audio locally on your machine (using Apple Silicon MPS, CUDA, or CPU):

1. Place your target 15-second reference audio file (`.wav`) into the `custom_prompts/` directory. Be sure the audio contains clean speech of the target voice.
2. Open `local_inference.py` in your text editor.
3. Update the `ref_audio_path` variable to point to your reference `.wav` file.
4. Update the `ref_text` variable with the **exact** Tamil text that is spoken in your reference audio.
5. In the `sentences` list at the bottom of the script, provide the Tamil text you want the model to generate, along with the desired output filename. For example:
   ```python
   sentences = [
       ('இவ் உயிர் மூலமாய் வெளிப்பட்டு ஏங்குவதன் ஒரே காரணம் நீயே நானாகும் நிலையை..', 'my_custom_output.wav')
   ]
   ```
6. Run the script:
   ```bash
   python local_inference.py
   ```

*Note: The script is configured to automatically download the fine-tuned model (`model_last.pt`) and custom vocabulary (`vocab.txt`) directly from the `kailasa-ngpt/IndicF5-Tamil-Finetuned` Hugging Face repository on its first run.*

---

## ⚠️ Important Notes on Tamil vs. Sanskrit Generation

- **Tamil Text (Native Script)**: The model was heavily fine-tuned on native Tamil characters. Supplying pure Tamil Unicode text to the prompt guarantees the best, most natural pronunciation.
- **Transliterated Sanskrit (English/Roman Letters)**: Do **not** pass English or ASCII transliterated text (like "sattva nurupa sarvasya") to the model. The underlying F5-TTS architecture cannot correctly synthesize transliterated Sanskrit text and will output heavily distorted or robotic audio. Always convert foreign phrases to native Tamil script before running inference.

---

## 👇 Original IndicF5 Project Information

*Below is the original upstream documentation for the base IndicF5 project.*

import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
    ref_audio_path="prompts/PAN_F_HAPPY_00001.wav",
    ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
```

## References

We would like to extend our gratitude to the authors of  **[F5-TTS](https://github.com/SWivid/F5-TTS)** for their invaluable contributions and inspiration to this work. Their efforts have played a crucial role in advancing  the field of text-to-speech synthesis.


## 📖 Citation
If you use **IndicF5** in your research or projects, please consider citing it:

### 🔹 BibTeX
```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}

