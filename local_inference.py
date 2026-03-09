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
ckpt_path = hf_hub_download(repo_id='ananthgv-usk/IndicF5-Tamil-Finetuned', filename='model_last.pt')
vocab_file = hf_hub_download(repo_id='ananthgv-usk/IndicF5-Tamil-Finetuned', filename='vocab.txt')

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
ref_audio_path = 'custom_prompts/tamil_male_reference_clipped.wav'
ref_text = 'அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை'

# Use the STANDARD upstream pipeline: preprocess_ref_audio_text -> infer_process
# This handles: audio clipping/normalization, ref_text punctuation, smart text chunking, cross-fading
print(f"Preprocessing reference audio: {ref_audio_path}")
ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text, show_info=print)
print(f"Processed ref_text: {ref_text_processed}")

os.makedirs('local_samples', exist_ok=True)

# Requested sentences
sentences = [
    ('வேதத்தின் சாரமே ஆகமம். வேதாந்தத்தின் உண்மையான சாரத்தை வழங்கும் ஞானம்தான் மிக உயர்ந்த சுபமான சித்தாந்தம்.', 'v4_vedham_full_local.wav'),
]

for gen_text, filename in sentences:
    print(f'\nGenerating: {filename}...')
    print(f'  Text: {gen_text}')
    
    # Use the standard infer_process pipeline (same as the upstream Gradio demo)
    # This handles: chunk_text with proper max_chars, infer_batch_process with cross-fading
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio_processed,
        ref_text_processed,
        gen_text,
        model_obj,
        vocoder,
        cross_fade_duration=0.15,
        speed=1.0,
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
