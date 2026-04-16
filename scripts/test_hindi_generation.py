#!/usr/bin/env python3
"""
Test script for Hindi generation from the multilingual model.
Based on base_eval.py and chat_cli.py

Usage:
    python -m scripts.test_hindi_generation --model-tag d6 --step 10
"""

import os
import argparse
import torch
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

def main():
    parser = argparse.ArgumentParser(description="Test Hindi generation from multilingual model")
    parser.add_argument('--model-tag', type=str, default='d6', help='Model tag (e.g., d6 for depth-6)')
    parser.add_argument('--step', type=int, default=None, help='Checkpoint step to load (default = last)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Initialize device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else nullcontext()

    # Load the model
    print0(f"\nLoading model {args.model_tag} at step {args.step}...")
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)

    # Create engine for generation
    engine = Engine(model, tokenizer)

    # Test prompts in both English and Hindi
    test_prompts = [    
        "नासा के पृथ्वी अवलोकन मिशन 1970 के दशक में टायरोस और निम्बस उपग्रहों के साथ शुरू हुए",
        "मुझे भारत के बारे में बताओ",
        "tell me about india",
        # # English prompts
        #             "The capital of France is",
        #     "The chemical symbol of gold is",
        #     "If yesterday was Friday, then tomorrow will be",
        #     "The opposite of hot is",
        #     "The planets of the solar system are:",
        #     "My favorite color is",
        #     "If 5*x + 3 = 13, then x is",
        # "The capital of India is",
        # "Machine learning is",
        # "Python programming",

        # # Hindi prompts
        # "भारत की राजधानी",
        # "नमस्ते, मैं",
        # "आज मौसम बहुत",
        # "हिंदी भाषा",
        # "मशीन लर्निंग",
        # "कंप्यूटर प्रोग्रामिंग",

        # # Mixed language prompts
        # "India का capital",
        # "Python एक programming",
        # "आज का weather",
    ]

    print0("\n" + "="*80)
    print0("Testing Hindi Generation")
    print0("="*80)

    for prompt in test_prompts:
        print0(f"\n{'='*40}")
        print0(f"Prompt: {prompt}")
        print0(f"{'='*40}")

        # Tokenize the prompt
        tokens = tokenizer.encode(prompt)

        # Add BOS token if tokenizer supports it
        if hasattr(tokenizer, 'get_bos_token_id'):
            bos = tokenizer.get_bos_token_id()
            tokens = [bos] + tokens

        # Generate text
        with autocast_ctx:
            generated_tokens, _ = engine.generate_batch(
                tokens,
                num_samples=1,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )

        # Decode and display
        generated_text = tokenizer.decode(generated_tokens[0])
        print0(f"Generated: {generated_text}")

    # Test unconditioned generation
    print0(f"\n{'='*80}")
    print0("Unconditioned Generation (random samples)")
    print0(f"{'='*80}\n")

    for i in range(3):
        # Start with just BOS token for unconditioned generation
        if hasattr(tokenizer, 'get_bos_token_id'):
            tokens = [tokenizer.get_bos_token_id()]
        else:
            tokens = []

        with autocast_ctx:
            generated_tokens, _ = engine.generate_batch(
                tokens,
                num_samples=1,
                max_tokens=args.max_tokens,
                temperature=1.0,  # Higher temperature for more diversity
                top_k=args.top_k
            )

        generated_text = tokenizer.decode(generated_tokens[0])
        print0(f"Sample {i+1}:\n{generated_text}\n{'-'*40}")

    # Cleanup
    compute_cleanup()
    print0("\nTest completed!")

if __name__ == "__main__":
    main()