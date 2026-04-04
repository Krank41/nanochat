"""
Count tokens in all shards using Sarvam tokenizer model with parallel processing.

This script uses multiprocessing to analyze multiple shards in parallel for faster processing.
python count_tokens_all_shards_parallel.py --workers 8
python count_tokens_all_shards_parallel.py --quick --tokenizer-type  sarvam
python count_tokens_all_shards_parallel.py --workers 8  --tokenizer-type  custom
"""

import os
import json
import time
import re
from typing import Dict, List, Tuple, Any
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial

def load_tokenizer(tokenizer_type: str = "sarvam", model_id: str = None) -> Tuple[Any, str, int]:
    """
    Load tokenizer based on type selection.

    Returns:
        tokenizer: The tokenizer object
        backend: 'huggingface' or 'sentencepiece'
        vocab_size: Size of vocabulary
    """
    if tokenizer_type == "sarvam":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id or "sarvamai/sarvam-1", use_fast=True)
        return tokenizer, "huggingface", len(tokenizer)

    elif tokenizer_type == "custom":
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(
            model_file='custom_tokenizer/nanochat_compatible_tokenizer.model'
        )
        return tokenizer, "sentencepiece", tokenizer.vocab_size()

    elif tokenizer_type == "gpt2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        return tokenizer, "huggingface", len(tokenizer)

    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def encode_text(tokenizer: Any, text: str, backend: str) -> List[int]:
    """
    Encode text using the appropriate tokenizer backend.
    """
    if backend == "huggingface":
        return tokenizer.encode(text, add_special_tokens=False)
    elif backend == "sentencepiece":
        return tokenizer.encode(text)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    # Check for Devanagari script (Hindi)
    devanagari_count = sum(1 for c in text[:500] if '\u0900' <= c <= '\u097F')
    # Check for ASCII/Latin (English)
    latin_count = sum(1 for c in text[:500] if 'a' <= c.lower() <= 'z')

    if devanagari_count > latin_count:
        return "Hindi"
    elif latin_count > 0:
        return "English"
    else:
        return "Unknown"

def count_words(text: str, language: str) -> int:
    """
    Count words in text based on language.

    For English: Split on whitespace and punctuation
    For Hindi: Split on whitespace and Devanagari punctuation
    """
    if language == "Hindi":
        # Hindi word boundaries: spaces and Devanagari punctuation
        words = re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+|\d+', text)
    else:
        # English and unknown: standard word tokenization
        words = re.findall(r'\b\w+\b', text)

    return len(words)

def analyze_shard_single(args: Tuple[str, str, str]) -> Dict:
    """Analyze a single shard file - to be run in parallel."""
    shard_path, tokenizer_type, model_id = args
    shard_name = os.path.basename(shard_path)

    print(f"Processing: {shard_name}")
    start_time = time.time()

    # Load tokenizer for this process
    tokenizer, backend, vocab_size = load_tokenizer(tokenizer_type, model_id)

    # Read the parquet file
    table = pq.read_table(shard_path)
    texts = table.column('text').to_pylist()
    doc_count = len(texts)

    # Process all documents in this shard
    total_tokens = 0
    total_chars = 0
    total_words = 0
    lang_stats = {"English": 0, "Hindi": 0, "Unknown": 0}
    lang_details = {
        "English": {"tokens": 0, "words": 0, "chars": 0},
        "Hindi": {"tokens": 0, "words": 0, "chars": 0},
        "Unknown": {"tokens": 0, "words": 0, "chars": 0}
    }
    doc_tokens = []
    doc_words = []

    for idx, text in enumerate(texts):
        if text is None or not isinstance(text, str):
            continue

        # Detect language
        lang = detect_language(text)
        lang_stats[lang] += 1

        # Count characters
        chars = len(text)
        total_chars += chars

        # Count words
        words = count_words(text, lang)
        total_words += words
        doc_words.append(words)

        # Tokenize
        tokens = encode_text(tokenizer, text, backend)
        num_tokens = len(tokens)
        total_tokens += num_tokens
        doc_tokens.append(num_tokens)

        # Update language-specific details
        lang_details[lang]["tokens"] += num_tokens
        lang_details[lang]["words"] += words
        lang_details[lang]["chars"] += chars

        # Progress indicator every 5000 docs
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"  {shard_name}: {idx + 1}/{doc_count} docs ({rate:.0f} docs/s)")

    # Calculate statistics
    avg_tokens_per_doc = total_tokens / doc_count if doc_count > 0 else 0
    avg_words_per_doc = total_words / doc_count if doc_count > 0 else 0
    avg_chars_per_doc = total_chars / doc_count if doc_count > 0 else 0
    tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
    fertility_rate = total_tokens / total_words if total_words > 0 else 0  # TOKENS PER WORD

    # Calculate language-specific fertility rates
    lang_fertility = {}
    for lang in ["English", "Hindi", "Unknown"]:
        if lang_details[lang]["words"] > 0:
            lang_fertility[lang] = round(lang_details[lang]["tokens"] / lang_details[lang]["words"], 4)
        else:
            lang_fertility[lang] = 0

    # File size
    file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)

    elapsed = time.time() - start_time
    print(f"✓ {shard_name}: {doc_count:,} docs, {total_tokens:,} tokens, fertility: {fertility_rate:.3f} tokens/word in {elapsed:.1f}s")

    return {
        "file": shard_name,
        "file_size_mb": round(file_size_mb, 2),
        "documents": doc_count,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_characters": total_chars,
        "fertility_rate": round(fertility_rate, 4),  # MAIN FERTILITY RATE
        "avg_tokens_per_doc": round(avg_tokens_per_doc, 2),
        "avg_words_per_doc": round(avg_words_per_doc, 2),
        "avg_chars_per_doc": round(avg_chars_per_doc, 2),
        "tokens_per_char": round(tokens_per_char, 4),
        "language_distribution": lang_stats,
        "language_details": lang_details,
        "language_fertility": lang_fertility,
        "min_doc_tokens": min(doc_tokens) if doc_tokens else 0,
        "max_doc_tokens": max(doc_tokens) if doc_tokens else 0,
        "processing_time": round(elapsed, 2)
    }

def analyze_multilingual_dataset_parallel(
    data_dir: str = "multi_lingual_data/multilingual-merged",
    tokenizer_type: str = "sarvam",
    model_id: str = None,
    output_file: str = "token_statistics.json",
    num_workers: int = None
):
    """Analyze all shards in the multilingual dataset using parallel processing."""

    if num_workers is None:
        num_workers = min(cpu_count() - 1, 8)  # Leave one CPU free, max 8

    print(f"Token Counter - Parallel Processing")
    print(f"{'='*50}")
    print(f"Tokenizer type: {tokenizer_type}")
    if model_id:
        print(f"Model ID: {model_id}")
    print(f"Parallel workers: {num_workers}")
    print(f"CPU cores available: {cpu_count()}")

    # Test load tokenizer once
    print(f"\nTesting tokenizer load...")
    tokenizer, backend, vocab_size = load_tokenizer(tokenizer_type, model_id)
    print(f"Tokenizer backend: {backend}")
    print(f"Tokenizer vocabulary size: {vocab_size}")
    del tokenizer  # Free memory

    # Get all shard files
    shard_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith('shard_') and f.endswith('.parquet')
    ])

    print(f"\nFound {len(shard_files)} shard files to analyze")

    # Load dataset metadata if available
    metadata_path = os.path.join(data_dir, "dataset_info.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"Dataset mixing strategy: {metadata.get('mixing_strategy', 'unknown')}")

    # Prepare arguments for parallel processing
    shard_args = [(shard, tokenizer_type, model_id) for shard in shard_files]

    print(f"\nProcessing {len(shard_files)} shards with {num_workers} workers...")
    print(f"{'='*50}")

    start_time = time.time()

    # Process all shards in parallel
    with Pool(num_workers) as pool:
        all_results = pool.map(analyze_shard_single, shard_args)

    elapsed_time = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"All shards processed in {elapsed_time:.1f} seconds!")

    # Calculate global statistics
    total_tokens_global = sum(r['total_tokens'] for r in all_results)
    total_words_global = sum(r['total_words'] for r in all_results)
    total_chars_global = sum(r['total_characters'] for r in all_results)
    total_docs_global = sum(r['documents'] for r in all_results)

    # Global fertility rate
    global_fertility_rate = total_tokens_global / total_words_global if total_words_global > 0 else 0

    lang_distribution_global = {"English": 0, "Hindi": 0, "Unknown": 0}
    lang_totals = {
        "English": {"tokens": 0, "words": 0, "chars": 0},
        "Hindi": {"tokens": 0, "words": 0, "chars": 0},
        "Unknown": {"tokens": 0, "words": 0, "chars": 0}
    }

    for result in all_results:
        for lang, count in result['language_distribution'].items():
            lang_distribution_global[lang] += count
        for lang in ["English", "Hindi", "Unknown"]:
            lang_totals[lang]["tokens"] += result["language_details"][lang]["tokens"]
            lang_totals[lang]["words"] += result["language_details"][lang]["words"]
            lang_totals[lang]["chars"] += result["language_details"][lang]["chars"]

    # Add metadata language info to results
    if metadata and 'shards' in metadata:
        for result in all_results:
            for shard_info in metadata['shards']:
                if shard_info['name'] == result['file']:
                    result['expected_language'] = shard_info.get('language', 'Unknown')
                    break

    # Sort results by shard name for better readability
    all_results.sort(key=lambda x: x['file'])

    global_stats = {
        "dataset_path": data_dir,
        "tokenizer_type": tokenizer_type,
        "tokenizer_model": model_id if model_id else f"default_{tokenizer_type}",
        "analysis_time_seconds": round(elapsed_time, 2),
        "processing_config": {
            "parallel_workers": num_workers,
            "total_cpu_cores": cpu_count()
        },
        "total_shards": len(shard_files),
        "total_documents": total_docs_global,
        "total_tokens": total_tokens_global,
        "total_words": total_words_global,
        "total_tokens_billions": round(total_tokens_global / 1e9, 3),
        "total_words_billions": round(total_words_global / 1e9, 3),
        "total_characters": total_chars_global,
        "total_characters_billions": round(total_chars_global / 1e9, 3),
        "fertility_rate": round(global_fertility_rate, 4),  # MAIN FERTILITY RATE (tokens per word)
        "avg_tokens_per_document": round(total_tokens_global / total_docs_global, 2) if total_docs_global > 0 else 0,
        "avg_words_per_document": round(total_words_global / total_docs_global, 2) if total_docs_global > 0 else 0,
        "avg_characters_per_document": round(total_chars_global / total_docs_global, 2) if total_docs_global > 0 else 0,
        "tokens_per_character": round(total_tokens_global / total_chars_global, 4) if total_chars_global > 0 else 0,
        "language_distribution": {
            "English": {
                "documents": lang_distribution_global["English"],
                "percentage": round(100 * lang_distribution_global["English"] / total_docs_global, 2) if total_docs_global > 0 else 0
            },
            "Hindi": {
                "documents": lang_distribution_global["Hindi"],
                "percentage": round(100 * lang_distribution_global["Hindi"] / total_docs_global, 2) if total_docs_global > 0 else 0
            },
            "Unknown": {
                "documents": lang_distribution_global["Unknown"],
                "percentage": round(100 * lang_distribution_global["Unknown"] / total_docs_global, 2) if total_docs_global > 0 else 0
            }
        },
        "language_fertility_rates": {
            "English": round(lang_totals["English"]["tokens"] / lang_totals["English"]["words"], 4) if lang_totals["English"]["words"] > 0 else 0,
            "Hindi": round(lang_totals["Hindi"]["tokens"] / lang_totals["Hindi"]["words"], 4) if lang_totals["Hindi"]["words"] > 0 else 0,
            "Unknown": round(lang_totals["Unknown"]["tokens"] / lang_totals["Unknown"]["words"], 4) if lang_totals["Unknown"]["words"] > 0 else 0
        },
        "per_shard_stats": all_results
    }

    # Save results to JSON
    output_path = os.path.join(data_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(global_stats, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"FINAL STATISTICS")
    print(f"{'='*50}")
    print(f"Total shards analyzed: {len(shard_files)}")
    print(f"Total documents: {total_docs_global:,}")
    print(f"Total words: {total_words_global:,} ({total_words_global/1e9:.3f}B)")
    print(f"Total tokens: {total_tokens_global:,} ({total_tokens_global/1e9:.3f}B)")
    print(f"Total characters: {total_chars_global:,} ({total_chars_global/1e9:.3f}B)")
    print(f"\n🔬 FERTILITY RATE: {global_stats['fertility_rate']:.4f} tokens/word")
    print(f"Average tokens per document: {global_stats['avg_tokens_per_document']:,.2f}")
    print(f"Average words per document: {global_stats['avg_words_per_document']:,.2f}")
    print(f"Average characters per document: {global_stats['avg_characters_per_document']:,.2f}")
    print(f"Tokens per character ratio: {global_stats['tokens_per_character']:.4f}")

    print(f"\nLanguage Distribution:")
    print(f"  English: {lang_distribution_global['English']:,} docs ({global_stats['language_distribution']['English']['percentage']:.1f}%)")
    print(f"    Fertility: {global_stats['language_fertility_rates']['English']:.4f} tokens/word")
    print(f"  Hindi: {lang_distribution_global['Hindi']:,} docs ({global_stats['language_distribution']['Hindi']['percentage']:.1f}%)")
    print(f"    Fertility: {global_stats['language_fertility_rates']['Hindi']:.4f} tokens/word")
    print(f"  Unknown: {lang_distribution_global['Unknown']:,} docs ({global_stats['language_distribution']['Unknown']['percentage']:.1f}%)")

    print(f"\nPerformance:")
    print(f"  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"  Avg time per shard: {elapsed_time/len(shard_files):.1f} seconds")
    print(f"  Processing rate: {total_docs_global/elapsed_time:.0f} docs/second")
    print(f"  Token rate: {total_tokens_global/elapsed_time/1e6:.2f}M tokens/second")

    print(f"\nResults saved to: {output_path}")

    # Print shard-level summary
    print(f"\nPer-Shard Summary:")
    print(f"{'Shard':<20} {'Docs':<10} {'Tokens':<15} {'Language Mix'}")
    print(f"{'-'*70}")
    for result in all_results[:10]:  # Show first 10
        lang_dist = result['language_distribution']
        lang_mix = f"En:{lang_dist['English']} Hi:{lang_dist['Hindi']}"
        print(f"{result['file']:<20} {result['documents']:<10,} {result['total_tokens']:<15,} {lang_mix}")
    if len(all_results) > 10:
        print(f"... and {len(all_results) - 10} more shards")

    return global_stats

def analyze_sample_fast(
    data_dir: str = "multi_lingual_data/multilingual-merged",
    tokenizer_type: str = "sarvam",
    model_id: str = None,
    sample_size: int = 100
):
    """Quick sample analysis for testing - analyze only a few documents per shard."""

    print(f"Quick Sample Analysis")
    print(f"{'='*50}")

    tokenizer, backend, vocab_size = load_tokenizer(tokenizer_type, model_id)
    print(f"Tokenizer: {tokenizer_type} (vocab size: {vocab_size})")

    # Get first few shards
    shard_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith('shard_') and f.endswith('.parquet')
    ])[:3]

    total_tokens = 0
    total_chars = 0
    total_docs = 0

    for shard_path in shard_files:
        print(f"\n{os.path.basename(shard_path)}:")
        table = pq.read_table(shard_path)
        texts = table.column('text').to_pylist()[:sample_size]

        shard_tokens = 0
        shard_chars = 0

        for text in texts:
            if text:
                tokens = encode_text(tokenizer, text, backend)
                shard_tokens += len(tokens)
                shard_chars += len(text)

        total_tokens += shard_tokens
        total_chars += shard_chars
        total_docs += len(texts)

        print(f"  Sample {len(texts)} docs: {shard_tokens:,} tokens, {shard_chars:,} chars")
        print(f"  Avg tokens/doc: {shard_tokens/len(texts):.0f}")
        print(f"  Tokens/char: {shard_tokens/shard_chars:.3f}")

    print(f"\nTotal sample stats:")
    print(f"  Docs: {total_docs}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Estimated total tokens (scaled): {total_tokens * (len(shard_files) / 3) * (50000 / sample_size) / 1e9:.2f}B")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count tokens in multilingual shards (parallel)")
    parser.add_argument("--data-dir",
                       default="multi_lingual_data/multilingual-merged",
                       help="Directory containing the shards")
    parser.add_argument("--tokenizer-type",
                       default="sarvam",
                       choices=["sarvam", "custom", "gpt2"],
                       help="Type of tokenizer to use (default: sarvam)")
    parser.add_argument("--model-id",
                       default=None,
                       help="Specific model ID to use (optional, uses defaults for each tokenizer type)")
    parser.add_argument("--output",
                       default="token_statistics.json",
                       help="Output JSON file name")
    parser.add_argument("--workers",
                       type=int,
                       default=None,
                       help="Number of parallel workers (default: auto)")
    parser.add_argument("--quick",
                       action="store_true",
                       help="Run quick sample analysis")

    args = parser.parse_args()

    if args.quick:
        analyze_sample_fast(args.data_dir, args.tokenizer_type, args.model_id)
    else:
        analyze_multilingual_dataset_parallel(
            args.data_dir,
            args.tokenizer_type,
            args.model_id,
            args.output,
            num_workers=args.workers
        )