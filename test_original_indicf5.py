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

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using compute device: {device}")

print("Downloading ORIGINAL F5-TTS Base model from Hugging Face...")
repo_id = 'SWivid/F5-TTS'
ckpt_path = hf_hub_download(repo_id=repo_id, filename='F5TTS_Base/model_1200000.safetensors')
vocab_file = hf_hub_download(repo_id=repo_id, filename='F5TTS_Base/vocab.txt')

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

# Set up reference audio from line 38/39
ref_audio_path = 'custom_prompts/sample_00005_15s.wav'
ref_text = 'செய்தி மகா கைலாசத்திருந்து அழுந்து கேளுங்கள். பறமாத்வைதம் எனும் முழ்மை நிலை ஏளிமையான புரிதலாள் மட்டுமே நிகவுந்து விட சாத்தியம் உண்டு '

print(f"Preprocessing reference audio: {ref_audio_path}")
ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text, show_info=print)
print(f"Processed ref_text: {ref_text_processed}")

os.makedirs('local_samples', exist_ok=True)

# Generate Tamil text on the base model
sentences = [
    ('இவ் உயிர் மூலமாய் வெளிப்பட்டு ஏங்குவதன் ஒரே காரணம் நீயே நானாகும் நிலையை, நானே நீயாகும் நிலையை இந்த பரமாத்வைதத்தை துடர்ந்து அனைவருக்கும் கொடுத்துக்கொண்டே கெழ்வி.', 'base_model_tamil_paramadvaitam.wav'),
]

for gen_text, filename in sentences:
    print(f'\nGenerating: {filename}...')
    print(f'  Text: {gen_text}')
    
    # Pure Tamil processing, speed matches 1.0
    effective_speed = 1.0
    print(f'  Using speed={effective_speed}')
    
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
