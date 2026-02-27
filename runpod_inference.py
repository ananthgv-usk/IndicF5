import os
import torch
import torchaudio
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, load_checkpoint, infer_batch_process

ref_audio = "/workspace/IndicF5/custom_prompts/tamil_male_reference_clipped.wav"
ref_text = "அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை"
gen_text = "ஸ்ரீ மங்கள அய்யா, நித்யானந்தம். நீங்கள் சாப்பிட்டீர்களா?"

ckpt_path = "/usr/local/lib/python3.11/ckpts/F5TTS_Base_vocos_pinyin_/workspace/IndicF5/custom_dataset_pinyin/model_last.pt"
vocab_file = "/workspace/IndicF5/custom_dataset_pinyin/vocab.txt"

vocoder = load_vocoder(is_local=False)
model_obj = load_model(model_cls=DiT, model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), mel_spec_type="vocos", vocab_file=vocab_file, ode_method="euler", use_ema=True, device="cuda")
model_obj = load_checkpoint(model_obj, ckpt_path, "cuda", use_ema=True)

audio, sr = torchaudio.load(ref_audio)
print(f"Original audio shape: {audio.shape}, sr: {sr}")

# To fix duration truncation: F5TTS `fix_duration` expects the TOTAL duration (ref_len + gen_len).
# We have a ~11s ref clip. Standard string length for Tamil is better than byte-length for duration checks.
# We'll allow roughly 10s of generation time for this sentence. Total = 21s.
calc_duration = 21.0 

try:
    final_wave, sr, spect = infer_batch_process(
        (audio, sr),
        ref_text,
        [gen_text],
        model_obj,
        vocoder,
        mel_spec_type="vocos",
        device="cuda",
        fix_duration=calc_duration
    )
    print("Success! Final wave shape:", final_wave.shape)
    torchaudio.save("/workspace/IndicF5/finetuned_test_4.wav", torch.tensor(final_wave).unsqueeze(0), sr)
except Exception as e:
    import traceback
    traceback.print_exc()
