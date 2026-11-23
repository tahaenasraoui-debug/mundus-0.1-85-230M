"""
Upload Mundus 0.1 Checkpoints and Model Card to Hugging Face Hub.
"""

import os
import sys
from huggingface_hub import HfApi, login

def main():
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
    user_repo = input(f"\nEnter repository ID [Default: {default_repo}]: ").strip()
    repo_id = user_repo if user_repo else default_repo

    print(f"\nTarget Repository: https://huggingface.co/{repo_id}")
    print("Creating repository if it does not exist...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # 3. Files to upload
    files_to_upload = [
        ("checkpoints/mundus_0.1_step_45000.pt", "mundus_0.1_step_45000.pt"),
        ("checkpoints/tokenizer.pt", "tokenizer.pt"),
        ("config.py", "config.py"),
        ("README.md", "README.md"),
        ("assets/dashboard.png", "assets/dashboard.png"),
    ]

    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            print(f"\nUploading {local_path} -> {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded {repo_path}")
        else:
            print(f"Skipping {local_path} (not found).")

    print("\n==================================================")
    print(f"All files uploaded successfully!")
    print(f"Model link: https://huggingface.co/{repo_id}")
    print("==================================================")


if __name__ == "__main__":
    main()
