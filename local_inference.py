import os
import torch
import numpy as np
import torchaudio
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, load_checkpoint,
    preprocess_ref_audio_text, infer_process,
)
from huggingface_hub import hf_hub_download

print("Setting up local inference on Apple Silicon (MPS)...")

# Force device to 'mps' if available, otherwise 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using compute device: {device}")

# Download the model and vocab from our Hugging Face repo
print("Downloading finetuned model and vocab from Hugging Face...")
ckpt_path = hf_hub_download(repo_id='kailasa-ngpt/IndicF5-Tamil-Finetuned', filename='model_last.pt')
vocab_file = hf_hub_download(repo_id='kailasa-ngpt/IndicF5-Tamil-Finetuned', filename='vocab.txt')

# Load components
print("Loading model and vocoder...")
vocoder = load_vocoder(is_local=False)
model_obj = load_model(
    model_cls=DiT, 
    model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), 
    mel_spec_type='vocos', 
    vocab_file=vocab_file, 
    ode_method='euler', 
    use_ema=True, 
    device=device
)
model_obj = load_checkpoint(model_obj, ckpt_path, device, use_ema=True)

# Set up reference audio
ref_audio_path = 'custom_prompts/sample_00005_15s.wav'
ref_text = 'செய்தி மகா கைலாசத்திருந்து அழுந்து கேளுங்கள். பறமாத்வைதம் எனும் முழ்மை நிலை ஏளிமையான புரிதலாள் மட்டுமே நிகவுந்து விட சாத்தியம் உண்டு '

# Use the STANDARD upstream pipeline: preprocess_ref_audio_text -> infer_process
# This handles: audio clipping/normalization, ref_text punctuation, smart text chunking, cross-fading
print(f"Preprocessing reference audio: {ref_audio_path}")
ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text, show_info=print)
print(f"Processed ref_text: {ref_text_processed}")

os.makedirs('local_samples', exist_ok=True)

# Requested sentences
sentences = [
    ('இவ் உயிர் மூலமாய் வெளிப்பட்டு ஏங்குவதன் ஒரே காரணம் நீயே நானாகும் நிலையை, நானே நீயாகும் நிலையை இந்த பரமாத்வைதத்தை துடர்ந்து அனைவருக்கும் கொடுத்துக்கொண்டே கெழ்வி.', 'finetuned_model_tamil_paramadvaitam.wav')
]

for gen_text, filename in sentences:
    print(f'\nGenerating: {filename}...')
    print(f'  Text: {gen_text}')
    
    # IMPORTANT: F5TTS calculates gen duration using UTF-8 byte length ratio:
    #   gen_duration ∝ ref_audio_len * (gen_bytes / ref_bytes)
    # Tamil ref_text = 357 bytes (3 bytes/char), ASCII gen_text = 103 bytes (1 byte/char)
    # This 3x byte ratio mismatch makes ASCII text get only 1/3 the needed duration.
    # Fix: use speed=0.35 to compensate (slower speed → longer duration).
    # For Tamil gen_text, use speed=1.0 (no compensation needed).
    is_ascii_text = all(ord(c) < 256 for c in gen_text if not c.isspace())
    effective_speed = 0.35 if is_ascii_text else 1.0
    print(f'  Using speed={effective_speed} ({"ASCII/Roman" if is_ascii_text else "Tamil"})')
    
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio_processed,
        ref_text_processed,
        gen_text,
        model_obj,
        vocoder,
        cross_fade_duration=0.15,
        speed=effective_speed,
        device=device,
    )
    
    # Normalize audio volume
    wave = np.array(final_wave, dtype=np.float32)
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.95
        
    out_path = f'local_samples/{filename}'
    torchaudio.save(out_path, torch.tensor(wave).unsqueeze(0), final_sample_rate)
    print(f'Done! Saved {out_path} ({wave.shape[0]/final_sample_rate:.1f}s)')
