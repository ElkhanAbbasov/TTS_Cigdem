"""Check what's stored in a checkpoint file and why it's so large"""
import torch
import os

checkpoint_path = "Models/Cigdem_TTS/checkpoints/epoch_2nd_00001.pth"

if os.path.exists(checkpoint_path):
    # Get file size
    size_bytes = os.path.getsize(checkpoint_path)
    size_gb = size_bytes / (1024**3)
    
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"💾 File size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    print(f"\n{'='*60}")
    
    # Load and analyze
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'net' in checkpoint:
        params = checkpoint['net']
    else:
        params = checkpoint
    
    print(f"\n📋 Components stored in checkpoint:")
    total_params = 0
    
    for key in params.keys():
        component = params[key]
        if isinstance(component, dict):
            # Count parameters in this component
            num_params = sum(p.numel() for p in component.values() if torch.is_tensor(p))
            num_tensors = sum(1 for p in component.values() if torch.is_tensor(p))
            
            # Estimate size (float32 = 4 bytes)
            size_mb = (num_params * 4) / (1024**2)
            
            total_params += num_params
            
            print(f"\n  🔹 {key}:")
            print(f"     - Parameters: {num_params:,}")
            print(f"     - Tensors: {num_tensors}")
            print(f"     - Size: ~{size_mb:.1f} MB")
    
    print(f"\n{'='*60}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Estimated size: ~{(total_params * 4) / (1024**3):.2f} GB")
    
    # Check if optimizer state is included
    if 'optimizer' in checkpoint or 'scheduler' in checkpoint:
        print(f"\n⚠️  Checkpoint includes optimizer state (doubles size!)")
        print(f"   Optimizer stores gradients + momentum for each parameter")
    
    print(f"\n{'='*60}")
    print(f"\n💡 Why so large?")
    print(f"   1. Multiple large neural networks:")
    print(f"      - BERT encoder (~110M params)")
    print(f"      - HiFiGAN decoder (~14M params)")
    print(f"      - Style encoder, discriminators, etc.")
    print(f"   2. Each stored in float32 (4 bytes per param)")
    print(f"   3. If optimizer included: 3x larger (param + gradient + momentum)")
    
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
