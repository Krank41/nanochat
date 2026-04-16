"""
Merge English and Hindi shards into a unified multilingual dataset for HuggingFace upload.

This script:
1. Combines shards from both languages
2. Renames them with unique indices
3. Optionally shuffles the shard order for better mixing
4. Prepares for HuggingFace upload


 export HF_TOKEN=''

 python merge_and_upload_multilingual.py --strategy random  --upload --repo-id "karana657/multilingual-nanochat"



"""

import os
import shutil
import random
from typing import List, Tuple
from huggingface_hub import HfApi
import argparse

def get_shard_info(shard_path: str) -> dict:
    """Get information about a shard file."""
    size_mb = os.path.getsize(shard_path) / (1024 * 1024)
    return {
        'path': shard_path,
        'size_mb': size_mb,
        'name': os.path.basename(shard_path)
    }

def merge_multilingual_shards(
    english_dir: str = "english-data",
    hindi_dir: str = "hindi-data-processed",
    output_dir: str = "multilingual-merged",
    mixing_strategy: str = "interleave",
    upload_to_hf: bool = False,
    hf_repo_id: str = None
):
    """
    Merge English and Hindi shards into a unified dataset.

    Args:
        english_dir: Directory containing English shards
        hindi_dir: Directory containing Hindi shards
        output_dir: Output directory for merged shards
        mixing_strategy: How to mix the shards
            - "interleave": Alternate between English and Hindi
            - "random": Randomly shuffle all shards
            - "sequential": All English first, then Hindi
            - "ratio": Custom ratio (e.g., 2:1 English:Hindi)
        upload_to_hf: Whether to upload to HuggingFace
        hf_repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all shard files
    english_shards = sorted([
        os.path.join(english_dir, f)
        for f in os.listdir(english_dir)
        if f.startswith('shard_') and f.endswith('.parquet')
    ])

    hindi_shards = sorted([
        os.path.join(hindi_dir, f)
        for f in os.listdir(hindi_dir)
        if f.startswith('shard_') and f.endswith('.parquet')
    ])

    print(f"Found {len(english_shards)} English shards")
    print(f"Found {len(hindi_shards)} Hindi shards")
    print(f"Total shards to merge: {len(english_shards) + len(hindi_shards)}")

    # Calculate total sizes
    english_size = sum(os.path.getsize(f) / (1024**3) for f in english_shards)  # GB
    hindi_size = sum(os.path.getsize(f) / (1024**3) for f in hindi_shards)  # GB

    print(f"\nDataset sizes:")
    print(f"  English: {english_size:.2f} GB")
    print(f"  Hindi: {hindi_size:.2f} GB")
    print(f"  Total: {english_size + hindi_size:.2f} GB")

    # Prepare merged shard list based on strategy
    merged_shards = []

    if mixing_strategy == "interleave":
        print("\nUsing INTERLEAVE strategy: alternating English and Hindi shards")
        # Alternate between English and Hindi
        max_len = max(len(english_shards), len(hindi_shards))
        for i in range(max_len):
            if i < len(english_shards):
                merged_shards.append(('en', english_shards[i]))
            if i < len(hindi_shards):
                merged_shards.append(('hi', hindi_shards[i]))

    elif mixing_strategy == "random":
        print("\nUsing RANDOM strategy: fully shuffling all shards")
        # Add all shards and shuffle randomly
        merged_shards = [('en', s) for s in english_shards] + [('hi', s) for s in hindi_shards]
        random.seed(42)  # For reproducibility
        random.shuffle(merged_shards)

    elif mixing_strategy == "sequential":
        print("\nUsing SEQUENTIAL strategy: English first, then Hindi")
        # All English first, then all Hindi
        merged_shards = [('en', s) for s in english_shards] + [('hi', s) for s in hindi_shards]

    elif mixing_strategy.startswith("ratio"):
        # Custom ratio like "ratio:2:1" for 2 English to 1 Hindi
        parts = mixing_strategy.split(":")
        if len(parts) == 3:
            en_ratio = int(parts[1])
            hi_ratio = int(parts[2])
            print(f"\nUsing RATIO strategy: {en_ratio} English : {hi_ratio} Hindi")

            en_idx, hi_idx = 0, 0
            while en_idx < len(english_shards) or hi_idx < len(hindi_shards):
                # Add English shards according to ratio
                for _ in range(en_ratio):
                    if en_idx < len(english_shards):
                        merged_shards.append(('en', english_shards[en_idx]))
                        en_idx += 1
                # Add Hindi shards according to ratio
                for _ in range(hi_ratio):
                    if hi_idx < len(hindi_shards):
                        merged_shards.append(('hi', hindi_shards[hi_idx]))
                        hi_idx += 1
        else:
            print("Invalid ratio format. Using interleave instead.")
            mixing_strategy = "interleave"

    # Copy/rename shards to output directory
    print(f"\nCopying and renaming {len(merged_shards)} shards to {output_dir}")

    metadata = {
        'shards': [],
        'total_shards': len(merged_shards),
        'english_shards': len(english_shards),
        'hindi_shards': len(hindi_shards),
        'mixing_strategy': mixing_strategy
    }

    for idx, (lang, shard_path) in enumerate(merged_shards):
        # Create new shard name with sequential numbering
        new_shard_name = f"shard_{idx:05d}.parquet"
        new_shard_path = os.path.join(output_dir, new_shard_name)

        # Copy the file
        shutil.copy2(shard_path, new_shard_path)

        # Get shard info
        size_mb = os.path.getsize(new_shard_path) / (1024 * 1024)

        # Add to metadata
        metadata['shards'].append({
            'index': idx,
            'name': new_shard_name,
            'language': 'English' if lang == 'en' else 'Hindi',
            'size_mb': round(size_mb, 2),
            'original_file': os.path.basename(shard_path)
        })

        # Progress update
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(merged_shards)} shards...")

    print(f"\nSuccessfully merged {len(merged_shards)} shards!")

    # Save metadata
    import json
    metadata_path = os.path.join(output_dir, "dataset_info.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Create README for the dataset
    readme_content = f"""# Multilingual Dataset (English + Hindi)

## Dataset Description
This dataset contains text data in English and Hindi, prepared for language model training.

## Statistics
- **Total Shards**: {len(merged_shards)}
- **English Shards**: {len(english_shards)}
- **Hindi Shards**: {len(hindi_shards)}
- **Total Size**: {english_size + hindi_size:.2f} GB
- **Mixing Strategy**: {mixing_strategy}

## Shard Format
- Format: Parquet files with zstd compression
- Schema: Single 'text' column containing the text data
- Row Group Size: 1024 documents per row group
- Compression: zstd level 3

## Language Distribution
- English: {english_size:.2f} GB (~{100*len(english_shards)/(len(english_shards)+len(hindi_shards)):.1f}% of shards)
- Hindi: {hindi_size:.2f} GB (~{100*len(hindi_shards)/(len(english_shards)+len(hindi_shards)):.1f}% of shards)

## Usage
```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("parquet", data_files="*.parquet")

# Load specific shards
dataset = load_dataset("parquet", data_files=["shard_00000.parquet", "shard_00001.parquet"])
```

## Mixing Strategies
- **interleave**: Shards alternate between English and Hindi
- **random**: All shards are randomly shuffled
- **sequential**: All English shards first, then all Hindi shards
- **ratio:X:Y**: X English shards for every Y Hindi shards
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Created README at {readme_path}")

    # Upload to HuggingFace if requested
    if upload_to_hf and hf_repo_id:
        print(f"\nUploading to HuggingFace: {hf_repo_id}")
        print("Note: Make sure you have set your HF_TOKEN environment variable")

        try:
            token = os.getenv("HF_TOKEN")
            if not token:
                print("ERROR: HF_TOKEN environment variable not set")
                print("Please set it with: export HF_TOKEN='your_token_here'")
                return

            api = HfApi(token=token)

            # Create the repository if it doesn't exist
            try:
                api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
                print(f"Repository {hf_repo_id} ready")
            except Exception as e:
                print(f"Note: {e}")

            # Upload the folder
            print("Starting upload... This may take a while for large datasets")
            api.upload_folder(
                folder_path=output_dir,
                repo_id=hf_repo_id,
                repo_type="dataset",
            )
            print(f"Successfully uploaded to https://huggingface.co/datasets/{hf_repo_id}")

        except Exception as e:
            print(f"Upload failed: {e}")
            print("You can manually upload later using the huggingface-cli")

    print("\nDone! Your merged dataset is ready at:", output_dir)

    # Print sample loading code
    print("\nTo use this dataset locally:")
    print(f"```python")
    print(f"from datasets import load_dataset")
    print(f'dataset = load_dataset("parquet", data_dir="{output_dir}")')
    print(f"```")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multilingual shards for HuggingFace")
    parser.add_argument("--english-dir", default="english-data")
    parser.add_argument("--hindi-dir", default="hindi-data-processed")
    parser.add_argument("--output-dir", default="multilingual-merged")
    parser.add_argument("--strategy", default="interleave",
                       choices=["interleave", "random", "sequential"],
                       help="Mixing strategy for shards")
    parser.add_argument("--ratio", type=str, help="Custom ratio like '2:1' for 2 English to 1 Hindi")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repo ID (e.g., username/dataset-name)")

    args = parser.parse_args()

    # Handle custom ratio
    strategy = args.strategy
    if args.ratio:
        strategy = f"ratio:{args.ratio}"

    merge_multilingual_shards(
        english_dir=args.english_dir,
        hindi_dir=args.hindi_dir,
        output_dir=args.output_dir,
        mixing_strategy=strategy,
        upload_to_hf=args.upload,
        hf_repo_id=args.repo_id
    )