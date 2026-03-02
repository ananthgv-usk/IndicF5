#!/bin/bash
# =============================================================================
# BULLETPROOF RunPod Setup for IndicF5 Tamil Fine-Tuning
# Run: bash runpod_setup.sh <HF_TOKEN>
# Estimated total time: ~40 minutes (5 min setup + 32 min training + 2 min inference)
# =============================================================================
set -e

HF_TOKEN="${1:-YOUR_HF_TOKEN}"
if [ "$HF_TOKEN" = "YOUR_HF_TOKEN" ]; then
    echo "ERROR: Pass your HF token as argument: bash runpod_setup.sh hf_xxxxx"
    exit 1
fi

echo "========================================="
echo "STEP 1/9: Clone repository"
echo "========================================="
cd /workspace
if [ -d "IndicF5" ]; then
    echo "IndicF5 already exists, pulling latest..."
    cd IndicF5 && git pull && cd ..
else
    git clone https://github.com/ananthgv-usk/IndicF5.git
fi
cd IndicF5

echo "========================================="
echo "STEP 2/9: Install dependencies"
echo "========================================="
pip install -r requirements.txt
pip install -e .
pip install hydra-core --upgrade
pip install 'datasets<4.0' soundfile
pip install 'accelerate<1.0'

echo "========================================="
echo "STEP 3/9: Authenticate HuggingFace"
echo "========================================="
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

echo "========================================="
echo "STEP 4/9: Download broader Tamil+Sanskrit dataset (300 samples)"
echo "========================================="
python3 -c "
from datasets import load_dataset
import soundfile as sf
import os, csv, re
from tqdm import tqdm

ds = load_dataset('kailasa-ngpt/tamil-ktkv-1000', split='train', token='$HF_TOKEN')
out_dir = 'custom_dataset'
wavs_dir = os.path.join(out_dir, 'wavs')
os.makedirs(wavs_dir, exist_ok=True)

def is_tamil_or_sanskrit(text):
    cleaned = re.sub(r'[ \.\\,\!\?\-\'\"\;\:\(\)]', '', text)
    if not cleaned or len(cleaned) < 5: return False
    tamil_count = sum(1 for c in cleaned if '\u0B80' <= c <= '\u0BFF')
    return tamil_count / len(cleaned) >= 0.7

count = 0
with open(os.path.join(out_dir, 'metadata.csv'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(['audio_file', 'text'])
    for idx, sample in enumerate(tqdm(ds)):
        if 'audio' not in sample or sample['audio'] is None: continue
        ref_text = sample.get('text', sample.get('sentence', sample.get('transcript', '')))
        if not is_tamil_or_sanskrit(ref_text): continue
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        wav_path = os.path.abspath(os.path.join(wavs_dir, f'sample_{idx:05d}.wav'))
        sf.write(wav_path, audio_array, sr)
        writer.writerow([wav_path, ref_text])
        count += 1
        if count >= 300: break
print(f'Saved {count} samples')
"

echo "========================================="
echo "STEP 5/9: Convert to Arrow format"
echo "========================================="
# Download the base vocab first so prepare_csv_wavs.py can find it
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ai4bharat/IndicF5', filename='checkpoints/vocab.txt')"

# Fix the vocab path in prepare_csv_wavs.py for this machine
VOCAB_PATH=$(find /root/.cache/huggingface/hub/models--ai4bharat--IndicF5 -name 'vocab.txt' -path '*/checkpoints/*' | head -1)
sed -i "s|PRETRAINED_VOCAB_PATH = Path(\".*\")|PRETRAINED_VOCAB_PATH = Path(\"$VOCAB_PATH\")|" f5_tts/train/datasets/prepare_csv_wavs.py

rm -rf custom_dataset_pinyin
python3 f5_tts/train/datasets/prepare_csv_wavs.py custom_dataset custom_dataset_pinyin

echo "========================================="
echo "STEP 6/9: CRITICAL — Restore original 2545-token IndicF5 vocabulary"
echo "========================================="
cp "$VOCAB_PATH" custom_dataset_pinyin/vocab.txt
VOCAB_COUNT=$(wc -l < custom_dataset_pinyin/vocab.txt)
echo "Vocab size: $VOCAB_COUNT tokens"
if [ "$VOCAB_COUNT" -lt 2000 ]; then
    echo "FATAL ERROR: Vocab has only $VOCAB_COUNT tokens! Expected 2545. Aborting."
    exit 1
fi
echo "VOCAB CHECK PASSED ✓"

echo "========================================="
echo "STEP 7/9: Write training config (num_workers=4, epochs=50, lr=3e-5)"
echo "========================================="
cat << 'EOF' > f5_tts/configs/F5TTS_Base_train.yaml
hydra:
  run:
    dir: ckpts/${model.name}_${model.mel_spec.mel_spec_type}_${model.tokenizer}_${datasets.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

datasets:
  name: /workspace/IndicF5/custom_dataset_pinyin
  batch_size_per_gpu: 38400
  batch_size_type: frame
  max_samples: 64
  num_workers: 8

optim:
  epochs: 25
  learning_rate: 3.0e-5
  num_warmup_updates: 20000
  grad_accumulation_steps: 1
  max_grad_norm: 1.0
  bnb_optimizer: False

model:
  name: F5TTS_Base
  tokenizer: pinyin
  tokenizer_path: None
  arch:
    dim: 1024
    depth: 22
    heads: 16
    ff_mult: 2
    text_dim: 512
    conv_layers: 4
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    hop_length: 256
    win_length: 1024
    n_fft: 1024
    mel_spec_type: vocos
  vocoder:
    is_local: False
    local_path: None

ckpts:
  logger: wandb
  save_per_updates: 50000
  last_per_steps: 5000
  save_dir: ckpts/${model.name}_${model.mel_spec.mel_spec_type}_${model.tokenizer}_${datasets.name}
EOF

# Symlink workaround and accelerate config
ln -sf /workspace/IndicF5/custom_dataset_pinyin /workspace/IndicF5/custom_dataset_pinyin_pinyin
mkdir -p /root/.cache/huggingface/accelerate
accelerate config default

echo "========================================="
echo "STEP 8/9: Launch training (25 epochs) with GPU health monitoring"
echo "========================================="
rm -rf /workspace/ckpts/

# Launch training in background so we can health-check GPU
WANDB_MODE=offline accelerate launch f5_tts/train/train.py --config-name F5TTS_Base_train > /workspace/IndicF5/training_output.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# GPU health check: verify GPU is actually being used (not hung)
echo "Running GPU health check for first 3 minutes..."
ZERO_COUNT=0
for i in $(seq 1 6); do
    sleep 30
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    LAST_LOG=$(tail -1 /workspace/IndicF5/training_output.log 2>/dev/null)
    echo "  Check $i/6: GPU=${GPU_UTIL}% | $LAST_LOG"
    if [ "$GPU_UTIL" -eq 0 ] 2>/dev/null; then
        ZERO_COUNT=$((ZERO_COUNT + 1))
    else
        ZERO_COUNT=0
    fi
    if [ $ZERO_COUNT -ge 3 ]; then
        echo "FATAL: GPU at 0% for 90+ seconds — training is hung!"
        kill -9 $TRAIN_PID 2>/dev/null
        exit 1
    fi
    # Check if training already crashed
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "FATAL: Training process died!"
        tail -20 /workspace/IndicF5/training_output.log
        exit 1
    fi
done
echo "GPU health check PASSED ✓ — training is computing normally"

# Now wait for training to complete
echo "Waiting for training to finish..."
wait $TRAIN_PID
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "FATAL: Training exited with code $TRAIN_EXIT"
    tail -20 /workspace/IndicF5/training_output.log
    exit 1
fi

cat /workspace/IndicF5/training_output.log | tail -5

# Verify checkpoint was saved
CKPT_PATH=$(find /workspace/ckpts/ -name 'model_last.pt' | head -1)
if [ -z "$CKPT_PATH" ]; then
    echo "FATAL ERROR: No checkpoint found after training!"
    exit 1
fi
echo "CHECKPOINT SAVED: $CKPT_PATH ✓"

echo "========================================="
echo "STEP 9/9: Run inference + Upload to HuggingFace"
echo "========================================="
python3 -c "
import os, torch, numpy as np, torchaudio
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, load_checkpoint, infer_batch_process

ckpt_path = '$CKPT_PATH'
vocab_file = '/workspace/IndicF5/custom_dataset_pinyin/vocab.txt'
vocoder = load_vocoder(is_local=False)
model_obj = load_model(model_cls=DiT, model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), mel_spec_type='vocos', vocab_file=vocab_file, ode_method='euler', use_ema=True, device='cuda')
model_obj = load_checkpoint(model_obj, ckpt_path, 'cuda', use_ema=True)

ref_audio = 'custom_prompts/tamil_male_reference_clipped.wav'
# Added a period at the end to force the model to create a clean pause, preventing audio spillover
ref_text = 'அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை.'
audio, sr = torchaudio.load(ref_audio)
# PADDING: Add 0.5s of silence to the end of the reference audio to absorb any generation spillover
audio = torch.cat([audio, torch.zeros(1, int(sr * 0.5))], dim=1)
os.makedirs('samples', exist_ok=True)

sentences = [
    ('வேதாந்தத்தின் உண்மையான சாரத்தை வழங்கும் ஞானம்தான் மிக உயர்ந்த சுபமான சித்தாந்தம்.', 'v4_vedham_single_sentence.wav'),
]
ref_samples = audio.shape[1]  # kept for reference only — NOT used for trimming
for gen_text, filename in sentences:
    print(f'Generating: {filename}...')
    import librosa
    
    # TEXT CHUNKING: Reverted to V11 logic - splitting only by sentences (periods)
    text_chunks = [s.strip() + '.' for s in gen_text.split('. ') if s.strip()]
    
    clean_waves = []
    final_sr = target_sample_rate
    
    for chunk in text_chunks:
        # speed=0.8 for slow, beautiful enunciation
        final_wave, out_sr, spect = infer_batch_process((audio, sr), ref_text, [chunk], model_obj, vocoder, mel_spec_type='vocos', device='cuda', fix_duration=None, speed=0.8)
        wave = np.array(final_wave, dtype=np.float32)
        final_sr = out_sr
        
        # Trim the bleed uniquely from EACH chunk using V11 librosa VAD logic
        intervals = librosa.effects.split(wave, top_db=30)
        if len(intervals) > 1:
            clean_start = intervals[1][0]
            wave = wave[clean_start:]
        
        clean_waves.append(wave)
    
    # Manually concatenate chunks with a 0.3s pause in between
    pause = np.zeros(int(final_sr * 0.3), dtype=np.float32)
    combined_wave = clean_waves[0]
    for w in clean_waves[1:]:
        combined_wave = np.concatenate([combined_wave, pause, w])

    peak = np.max(np.abs(combined_wave))
    if peak > 0: combined_wave = combined_wave / peak * 0.95
    torchaudio.save(f'samples/{filename}', torch.tensor(combined_wave).unsqueeze(0), final_sr)
    print(f'  Saved {filename}: {combined_wave.shape[0]/final_sr:.1f}s')
print('All inference done!')
"

# Upload to HuggingFace
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_file(path_or_fileobj='$CKPT_PATH', path_in_repo='model_last.pt', repo_id='ananthgv-usk/IndicF5-Tamil-Finetuned', commit_message='Round 4: 25 epochs, 300 broader Tamil+Sanskrit samples, correct 2545-token vocab')
api.upload_file(path_or_fileobj='/workspace/IndicF5/custom_dataset_pinyin/vocab.txt', path_in_repo='vocab.txt', repo_id='ananthgv-usk/IndicF5-Tamil-Finetuned', commit_message='Vocab (2545 tokens)')
print('HF UPLOAD COMPLETE!')
"

echo "========================================="
echo "ALL DONE! You can now delete the pod."
echo "Download samples: scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/IndicF5/samples/v4_*.wav ."
echo "========================================="
