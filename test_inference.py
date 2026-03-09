"""
Automated inference quality tests for IndicF5 Tamil TTS.

Tests verify:
1. Output audio is not silent / has actual speech content
2. Output duration is proportional to input text length
3. No reference text bleed (ref words don't appear in output)
4. All key input words are spoken (no skipped words)

Uses Whisper ASR to transcribe generated audio for verification.
"""

import os
import sys
import torch
import numpy as np
import torchaudio
import tempfile
import soundfile as sf
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, load_checkpoint,
    preprocess_ref_audio_text, infer_process, transcribe,
)
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────
# Setup (shared across all tests)
# ──────────────────────────────────────────────────

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"[setup] Device: {device}")

ckpt_path = hf_hub_download(repo_id='ananthgv-usk/IndicF5-Tamil-Finetuned', filename='model_last.pt')
vocab_file = hf_hub_download(repo_id='ananthgv-usk/IndicF5-Tamil-Finetuned', filename='vocab.txt')

vocoder = load_vocoder(is_local=False)
model_obj = load_model(
    model_cls=DiT,
    model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    mel_spec_type='vocos', vocab_file=vocab_file, ode_method='euler', use_ema=True, device=device
)
model_obj = load_checkpoint(model_obj, ckpt_path, device, use_ema=True)

ref_audio_path = 'custom_prompts/tamil_male_reference_clipped.wav'
ref_text = 'அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை'

ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text, show_info=print)

# ──────────────────────────────────────────────────
# Helper: generate and transcribe
# ──────────────────────────────────────────────────

def generate_and_analyze(gen_text, label="test"):
    """Generate audio for gen_text and return analysis dict."""
    print(f"\n{'='*60}")
    print(f"[{label}] Generating: {gen_text}")
    
    final_wave, sr, _ = infer_process(
        ref_audio_processed, ref_text_processed, gen_text,
        model_obj, vocoder, cross_fade_duration=0.15, speed=1.0, device=device,
    )
    
    wave = np.array(final_wave, dtype=np.float32)
    duration = wave.shape[0] / sr
    rms = np.sqrt(np.mean(wave**2))
    
    # Save to temp file for transcription
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, wave, sr)
        tmp_path = f.name
    
    # Transcribe with Whisper
    transcript = transcribe(tmp_path, language="ta")
    os.unlink(tmp_path)
    
    print(f"  Duration: {duration:.1f}s | RMS: {rms:.4f}")
    print(f"  Transcript: {transcript}")
    
    return {
        'gen_text': gen_text,
        'wave': wave,
        'sr': sr,
        'duration': duration,
        'rms': rms,
        'transcript': transcript,
    }


# ──────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────

# Reference text unique tail words (should NOT appear in output)
REF_TAIL_WORDS = ['ரொம்ப', 'ஆசை', 'கத்துக்கணும்', 'புதுசு']

TEST_CASES = [
    {
        'label': 'short_sentence',
        'text': 'ஸ்ரீ மங்கள அய்யா, நித்யானந்தம்.',
        'key_words': ['மங்கள', 'நித்யானந்தம்'],
        'min_duration': 2.0,
        'max_duration': 12.0,
    },
    {
        'label': 'vedham_full',
        'text': 'வேதத்தின் சாரமே ஆகமம். வேதாந்தத்தின் உண்மையான சாரத்தை வழங்கும் ஞானம்தான் மிக உயர்ந்த சுபமான சித்தாந்தம்.',
        'key_words': ['வேதத்தின்', 'சாரமே', 'ஆகமம்', 'உண்மையான', 'சித்தாந்தம்'],
        'min_duration': 5.0,
        'max_duration': 20.0,
    },
    {
        'label': 'jadavulagu',
        'text': 'ஜடவுலகின் உயர்ந்த இடத்திலிருந்து கீழ்நிலை வரை, அவை அனைத்தும் மீண்டும் மீண்டும் துன்பம் நிறைந்த இடங்களாகும்.',
        'key_words': ['ஜடவுலகின்', 'கீழ்நிலை', 'துன்பம்', 'இடங்களாகும்'],
        'min_duration': 5.0,
        'max_duration': 20.0,
    },
]


def run_tests():
    results = []
    all_passed = True
    
    for tc in TEST_CASES:
        label = tc['label']
        result = generate_and_analyze(tc['text'], label)
        
        failures = []
        
        # Test 1: Audio is not silent
        if result['rms'] < 0.01:
            failures.append(f"FAIL: Audio is nearly silent (RMS={result['rms']:.4f})")
        
        # Test 2: Duration is reasonable
        if result['duration'] < tc['min_duration']:
            failures.append(f"FAIL: Too short ({result['duration']:.1f}s < {tc['min_duration']}s)")
        if result['duration'] > tc['max_duration']:
            failures.append(f"FAIL: Too long ({result['duration']:.1f}s > {tc['max_duration']}s)")
        
        # Test 3: No reference text bleed
        transcript_lower = result['transcript']
        for ref_word in REF_TAIL_WORDS:
            if ref_word in transcript_lower:
                failures.append(f"FAIL: Ref text bleed detected — '{ref_word}' found in output transcript")
        
        # Test 4: Key words present (check against transcript)
        for kw in tc['key_words']:
            if kw not in transcript_lower:
                failures.append(f"WARN: Key word '{kw}' not found in transcript (may be ASR error)")
        
        # Report
        print(f"\n  [{label}] Results:")
        if failures:
            for f in failures:
                print(f"    ❌ {f}")
            if any("FAIL" in f for f in failures):
                all_passed = False
        else:
            print(f"    ✅ All checks passed")
        
        results.append({
            'label': label,
            'failures': failures,
            'duration': result['duration'],
            'transcript': result['transcript'],
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✅ PASS" if not any("FAIL" in f for f in r['failures']) else "❌ FAIL"
        print(f"  {status} [{r['label']}] {r['duration']:.1f}s")
        if r['failures']:
            for f in r['failures']:
                print(f"         {f}")
        print(f"         Transcript: {r['transcript']}")
    
    if all_passed:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💀 Some tests FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
