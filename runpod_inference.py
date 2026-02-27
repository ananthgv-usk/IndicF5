import os
import torch
import torchaudio
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, load_checkpoint, infer_batch_process
from huggingface_hub import hf_hub_download

# Ref audio should exist securely in the repo's prompt folder, not hardcoded absolute path
ref_audio = "custom_prompts/tamil_male_reference_pure.wav"
ref_text = "அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை"
gen_text = "ஸ்ரீ மங்கள அய்யா, நித்யானந்தம். நீங்கள் சாப்பிட்டீர்களா?"

print("Downloading Fine-Tuned Model and Vocab from Hugging Face...")
repo_id = "ananthgv-usk/IndicF5-Tamil-Finetuned"
ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_last.pt")
vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.txt")

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

vocoder = load_vocoder(is_local=False)
model_obj = load_model(model_cls=DiT, model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), mel_spec_type="vocos", vocab_file=vocab_file, ode_method="euler", use_ema=True, device=device)
model_obj = load_checkpoint(model_obj, ckpt_path, device, use_ema=True)

audio, sr = torchaudio.load(ref_audio)
print(f"Original audio shape: {audio.shape}, sr: {sr}")

# To fix duration truncation: F5TTS `fix_duration` expects the TOTAL duration (ref_len + gen_len).
# Standard string length for Tamil is better than byte-length for duration checks.
# We'll allow roughly 10s of generation time for this sentence. Total = 21s.
try:
    final_wave, sr, spect = infer_batch_process(
        (audio, sr),
        ref_text,
        [gen_text],
        model_obj,
        vocoder,
        mel_spec_type="vocos",
        device=device,
        fix_duration=None
    )
    print("Success! Final wave shape:", final_wave.shape)
    
    # --- Post-processing: Normalize volume to prevent uneven tone ---
    import numpy as np
    wave = np.array(final_wave, dtype=np.float32)
    
    # 1. Peak normalization: scale so the loudest point hits 0.95
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.95
    
    # 2. RMS-based loudness leveling (target -20 dBFS)
    rms = np.sqrt(np.mean(wave ** 2))
    target_rms = 10 ** (-20 / 20)  # -20 dBFS
    if rms > 0:
        gain = target_rms / rms
        wave = wave * gain
        # Clip to prevent distortion
        wave = np.clip(wave, -1.0, 1.0)
    
    print(f"Normalized: peak={np.max(np.abs(wave)):.3f}, rms={np.sqrt(np.mean(wave**2)):.4f}")
    
    # Save safely to the local working directory (samples/ folder)
    os.makedirs("samples", exist_ok=True)
    torchaudio.save("samples/runpod_finetuned_test.wav", torch.tensor(wave).unsqueeze(0), sr)
except Exception as e:
    import traceback
    traceback.print_exc()
