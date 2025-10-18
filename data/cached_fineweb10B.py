#!/usr/bin/env python3
"""
Download the GPT-2 tokenized FineWeb10B dataset chunks from Hugging Face Hub.
Saves files to ./fineweb10B/ in the same directory as this script.
"""

import os
import sys
from huggingface_hub import hf_hub_download


def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), "fineweb10B")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, fname)
    if not os.path.exists(local_path):
        print(f"Downloading {fname}...")
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
        )
    else:
        print(f"{fname} already exists. Skipping.")


def main():
    # Download validation chunk
    get("fineweb_val_%06d.bin" % 0)

    # Download training chunks (1 to 50 inclusive = 5B tokens)
    num_chunks = 50
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)

    print("✅ All requested chunks downloaded.")


if __name__ == "__main__":
    main()
