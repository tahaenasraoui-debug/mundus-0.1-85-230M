"""
Upload Mundus 0.1 Checkpoints and Model Card to Hugging Face Hub.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login

from config import BASE_DIR, DEFAULT_CHECKPOINT_DIR, DEFAULT_TOKENIZER_PATH


def main():
    parser = argparse.ArgumentParser(description="Upload Mundus 0.1 checkpoints to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default=None, help="Target Hugging Face repository ID")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR / "mundus_0.1_step_45000.pt"),
        help="Path to world model checkpoint file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
        help="Path to VQ-VAE tokenizer checkpoint file",
    )
    args = parser.parse_args()

    print("==================================================")
    print("   Upload Mundus 0.1 to Hugging Face Hub")
    print("==================================================")

    # 1. Check / prompt for Hugging Face Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nPlease enter your Hugging Face Access Token with WRITE permission.")
        print("You can get or create one here: https://huggingface.co/settings/tokens\n")
        token = input("HF Token: ").strip()

    if not token:
        print("Error: No token provided. Aborting.")
        return

    # Log in
    login(token=token, add_to_git_credential=True)
    api = HfApi(token=token)

    # 2. Get target repository name
    default_repo = "Tahary/mundus-0.1-world-model"
    if args.repo_id:
        repo_id = args.repo_id
    else:
        user_repo = input(f"\nEnter repository ID [Default: {default_repo}]: ").strip()
        repo_id = user_repo if user_repo else default_repo

    print(f"\nTarget Repository: https://huggingface.co/{repo_id}")
    print("Creating repository if it does not exist...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # 3. Files to upload
    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)
    files_to_upload = [
        (ckpt_path, ckpt_path.name),
        (tok_path, tok_path.name),
        (BASE_DIR / "config.py", "config.py"),
        (BASE_DIR / "README.md", "README.md"),
        (BASE_DIR / "assets" / "dashboard.png", "assets/dashboard.png"),
    ]

    for local_path, repo_path in files_to_upload:
        if local_path.exists():
            print(f"\nUploading {local_path} -> {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded {repo_path}")
        else:
            print(f"Skipping {local_path} (not found).")

    print("\n==================================================")
    print("All files uploaded successfully!")
    print(f"Model link: https://huggingface.co/{repo_id}")
    print("==================================================")


if __name__ == "__main__":
    main()
