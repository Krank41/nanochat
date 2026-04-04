#!/usr/bin/env python3
"""
Ultra-minimal memory approach - samples BEFORE extraction to minimize memory use.
python train_tokenizer_minimal.py --vocab-size 32768 --output-prefix nanochat_compatible_tokenizer
"""

import os
import pandas as pd
import sentencepiece as spm
from pathlib import Path
import random
import tempfile
import gc

def extract_sample_data(english_dir="english_dataset", hindi_dir="hindi_dataset",
                        max_docs=400000, output_file="training_data.txt"):
    """
    Extract a balanced sample for tokenizer training.
    400K docs provides good coverage while being memory-safe.
    """

    print("="*60)
    print("MINIMAL MEMORY TOKENIZER TRAINING")
    print("="*60)

    # Get file lists
    english_files = sorted(Path(english_dir).glob("*.parquet"))
    hindi_files = sorted(Path(hindi_dir).glob("*.parquet"))

    print(f"Found {len(english_files)} English files")
    print(f"Found {len(hindi_files)} Hindi files")

    # We'll extract equal amounts from each language
    docs_per_language = max_docs // 2

    print(f"\nTarget: {docs_per_language:,} documents per language")
    print(f"Total: {max_docs:,} documents")

    with open(output_file, 'w', encoding='utf-8') as out_f:

        # Process English files
        print(f"\nExtracting {docs_per_language:,} English documents...")
        english_written = 0

        # Calculate how many docs to take per file
        docs_per_file = docs_per_language // len(english_files) + 1

        for file_idx, file_path in enumerate(english_files):
            if english_written >= docs_per_language:
                break

            try:
                # Read just this one file
                df = pd.read_parquet(file_path)

                # Find text column
                text_col = 'text'
                if text_col not in df.columns:
                    for col in ['content', 'document']:
                        if col in df.columns:
                            text_col = col
                            break

                # Sample from this file
                sample_size = min(docs_per_file, len(df), docs_per_language - english_written)
                if sample_size > 0:
                    sampled_df = df.sample(n=sample_size, random_state=42 + file_idx)

                    for text in sampled_df[text_col]:
                        if isinstance(text, str) and text.strip():
                            # Clean and write
                            clean_text = text.strip().replace('\n', ' ').replace('\r', ' ')
                            out_f.write(clean_text + '\n')
                            english_written += 1

                            if english_written % 10000 == 0:
                                print(f"  English: {english_written:,} documents")

                # Free memory immediately
                del df
                if sample_size > 0:
                    del sampled_df
                gc.collect()

            except Exception as e:
                print(f"  Error with {file_path.name}: {e}")

        print(f"  Total English: {english_written:,} documents")

        # Process Hindi files
        print(f"\nExtracting {docs_per_language:,} Hindi documents...")
        hindi_written = 0

        for file_path in hindi_files:
            if hindi_written >= docs_per_language:
                break

            try:
                # Read the Hindi file
                df = pd.read_parquet(file_path)

                # Find text column
                text_col = 'text'
                if text_col not in df.columns:
                    for col in ['content', 'document']:
                        if col in df.columns:
                            text_col = col
                            break

                # Sample from this file
                sample_size = min(docs_per_language, len(df))
                sampled_df = df.sample(n=sample_size, random_state=42)

                for text in sampled_df[text_col]:
                    if isinstance(text, str) and text.strip():
                        # Clean and write
                        clean_text = text.strip().replace('\n', ' ').replace('\r', ' ')
                        out_f.write(clean_text + '\n')
                        hindi_written += 1

                        if hindi_written % 10000 == 0:
                            print(f"  Hindi: {hindi_written:,} documents")

                        if hindi_written >= docs_per_language:
                            break

                # Free memory immediately
                del df
                del sampled_df
                gc.collect()

            except Exception as e:
                print(f"  Error with {file_path.name}: {e}")

        print(f"  Total Hindi: {hindi_written:,} documents")

        total_written = english_written + hindi_written
        print(f"\nTotal documents written: {total_written:,}")
        print(f"Language ratio - English: {english_written/total_written*100:.1f}%, Hindi: {hindi_written/total_written*100:.1f}%")

    return output_file, total_written

def train_tokenizer_minimal(vocab_size=32768, output_prefix="tokenizer"):
    """Train tokenizer with minimal memory usage."""

    # First extract a sample
    data_file, total_docs = extract_sample_data(max_docs=400000)

    print(f"\n{'='*60}")
    print(f"Training SentencePiece on {total_docs:,} documents...")
    print("This should complete without memory issues...")
    print("="*60)

    # Train with very conservative settings
    train_args = (
        f"--input={data_file} "
        f"--model_prefix={output_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=0.9995 "
        f"--max_sentence_length=16384 "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        f"--pad_piece=<pad> --unk_piece=<unk> --bos_piece=<|bos|> --eos_piece=</s> "
        f"--user_defined_symbols=<|user_start|>,<|user_end|>,<|assistant_start|>,<|assistant_end|>,<|python_start|>,<|python_end|>,<|output_start|>,<|output_end|> "
        f"--shuffle_input_sentence=true "
        f"--num_threads=4"  # Moderate threads for 400K docs
    )

    # Note: We're NOT using --input_sentence_size since we already limited the input

    spm.SentencePieceTrainer.train(train_args)

    # Clean up
    os.unlink(data_file)

    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"Model saved as: {output_prefix}.model")
    print(f"Vocab saved as: {output_prefix}.vocab")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-size', type=int, default=32768)
    parser.add_argument('--output-prefix', default='nanochat_compatible_tokenizer')
    args = parser.parse_args()

    train_tokenizer_minimal(
        vocab_size=args.vocab_size,
        output_prefix=args.output_prefix
    )