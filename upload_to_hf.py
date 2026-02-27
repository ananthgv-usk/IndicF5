import os
from huggingface_hub import HfApi

# Define repository details
repo_id = "ananthgv-usk/IndicF5-Tamil-Finetuned"
hf_token = "INSERT_HUGGINGFACE_TOKEN_HERE"  # Set to a write-enabled token

# Define path to the model weights on RunPod
checkpoint_dir = "/usr/local/lib/python3.11/ckpts/F5TTS_Base_vocos_pinyin_/workspace/IndicF5/custom_dataset_pinyin/"
model_path = os.path.join(checkpoint_dir, "model_last.pt")

print("Initializing Hugging Face API...")
api = HfApi(token=hf_token)

print(f"Creating repository {repo_id} if it doesn't exist...")
try:
    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
except Exception as e:
    print(f"Repo creation notice: {e}")

print(f"Uploading {model_path} to {repo_id}...")
try:
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model_last.pt",
        repo_id=repo_id,
        commit_message="Upload first successful Tamil fine-tuned weights (15 epochs, vocabulary restored)",
    )
    
    # Also attempt to upload the vocabulary file so the inference works off the shelf
    vocab_path = os.path.join(checkpoint_dir, "vocab.txt")
    if os.path.exists(vocab_path):
        api.upload_file(
            path_or_fileobj=vocab_path,
            path_in_repo="vocab.txt",
            repo_id=repo_id,
            commit_message="Upload full IndicF5 base vocabulary mappings",
        )
    
    print("SUCCESS! Model weights securely backed up to Hugging Face.")
except Exception as e:
    print(f"Upload Failed: {e}")
