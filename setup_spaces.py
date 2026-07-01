#!/usr/bin/env python3
"""
Helper script to set up model paths for HuggingFace Spaces ZeroGPU deployment.

This script helps you configure the environment variables needed for the Gradio app.
Run this interactively to set up your Space configuration.
"""

import os
from pathlib import Path


def setup_environment():
    """Interactive setup for environment variables."""
    print("\n" + "=" * 70)
    print("LTX-2 HuggingFace Spaces Setup Helper")
    print("=" * 70 + "\n")
    
    print("This helper will guide you through setting up environment variables")
    print("for your private HuggingFace Space with ZeroGPU.\n")
    
    paths = {}
    
    # Checkpoint path
    print("1️⃣  LTX-2 Model Checkpoint")
    print("   Download from: https://huggingface.co/Lightricks/LTX-2.3")
    print("   Recommended: ltx-2.3-22b-dev.safetensors (for full quality)")
    checkpoint = input("   Enter path to checkpoint: ").strip()
    if checkpoint:
        paths["LTX2_CHECKPOINT_PATH"] = checkpoint
    
    # Distilled LoRA path
    print("\n2️⃣  Distilled LoRA")
    print("   Download from: https://huggingface.co/Lightricks/LTX-2.3")
    print("   File: ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
    distilled_lora = input("   Enter path to distilled LoRA: ").strip()
    if distilled_lora:
        paths["LTX2_DISTILLED_LORA_PATH"] = distilled_lora
    
    # Spatial upsampler path
    print("\n3️⃣  Spatial Upsampler")
    print("   Download from: https://huggingface.co/Lightricks/LTX-2.3")
    print("   File: ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    upsampler = input("   Enter path to spatial upsampler: ").strip()
    if upsampler:
        paths["LTX2_SPATIAL_UPSAMPLER_PATH"] = upsampler
    
    # Gemma root path
    print("\n4️⃣  Gemma Text Encoder")
    print("   Download from: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized")
    print("   Download all files from that repository")
    gemma = input("   Enter path to Gemma root directory: ").strip()
    if gemma:
        paths["LTX2_GEMMA_ROOT"] = gemma
    
    print("\n" + "=" * 70)
    print("📋 HuggingFace Space Settings Configuration")
    print("=" * 70 + "\n")
    
    print("Add these to your Space's Settings → Variables and secrets:\n")
    for key, value in paths.items():
        print(f"{key}={value}")
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70 + "\n")
    
    print("Next steps:")
    print("1. Go to your private HuggingFace Space settings")
    print("2. Click 'Variables and secrets' → 'Add secret'")
    print("3. Add each variable above (they'll be hidden and not in git)")
    print("4. Set Hardware to 'ZeroGPU' in Space settings")
    print("5. The app.py will automatically use these variables\n")


if __name__ == "__main__":
    setup_environment()
