"""Quick test to check if model produces sound"""
import torch
import torchaudio

# Simple check: Load checkpoint and verify it has data
checkpoint_path = "Models/Cigdem_TTS/epoch_2nd_00001.pth"

print("="*60)
print("CHECKPOINT VALIDATION TEST")
print("="*60)

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'net' in checkpoint:
        params = checkpoint['net']
    else:
        params = checkpoint
    
    print(f"\n✅ Checkpoint loaded successfully!")
    print(f"\n📦 Model components found:")
    for key in params.keys():
        component = params[key]
        if isinstance(component, dict):
            num_params = sum(p.numel() for p in component.values() if torch.is_tensor(p))
            has_data = any((torch.is_tensor(p) and not torch.isnan(p).any() and not torch.isinf(p).any()) 
                          for p in component.values() if torch.is_tensor(p))
            print(f"  • {key}: {len(component)} params, Valid: {'✅' if has_data else '❌'}")
    
    # Check decoder specifically (critical for audio generation)
    if 'decoder' in params:
        decoder_params = params['decoder']
        print(f"\n🔊 Decoder check:")
        
        # Sample a few decoder parameters
        sample_keys = list(decoder_params.keys())[:3]
        for key in sample_keys:
            param = decoder_params[key]
            if torch.is_tensor(param):
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                mean_val = param.abs().mean().item()
                print(f"  • {key}: mean={mean_val:.6f}, NaN={has_nan}, Inf={has_inf}")
        
        if mean_val > 0 and not has_nan and not has_inf:
            print(f"\n✅ Decoder appears healthy! Should produce sound.")
        else:
            print(f"\n❌ Decoder has issues! May produce silent output.")
    
    print("\n" + "="*60)
    print("Now run: python test_cigdem.py")
    print("(Make sure to press Ctrl+C when prompted for custom text)")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
