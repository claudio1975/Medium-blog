#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and convert to GGUF.
Fixed for Windows encoding issues.
"""

import os
import torch
from pathlib import Path

ADAPTER_DIR = "qwen3-medical"
BASE_MODEL = "unsloth/Qwen3-0.6B"
MERGED_DIR = "qwen3-medical-merged"

def main():
    print("=" * 60)
    print("Merging LoRA adapter with base model")
    print("=" * 60)
    
    print(f"\nBase model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Output: {MERGED_DIR}")
    
    # Check if already merged
    if Path(MERGED_DIR).exists() and (Path(MERGED_DIR) / "config.json").exists():
        print(f"\n[OK] Merged model already exists at {MERGED_DIR}")
        print("Delete the folder if you want to re-merge.")
        return
    
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print(f"[3/4] Loading LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("[3/4] Merging weights...")
    model = model.merge_and_unload()
    
    print(f"[4/4] Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    
    print("\n" + "=" * 60)
    print("SUCCESS! Merged model saved to:", MERGED_DIR)
    print("=" * 60)
    
    print("\nNext step - convert to GGUF:")
    print("  python llama.cpp/convert_hf_to_gguf.py qwen3-medical-merged --outfile qwen3-medical.gguf --outtype q8_0")


if __name__ == "__main__":
    main()
