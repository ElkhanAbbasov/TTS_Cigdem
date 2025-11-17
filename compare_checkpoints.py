"""
Compare multiple checkpoints - test epoch 4, 9, and 14
Generates the same text with all three models for comparison
"""

import torch
import torchaudio
import os
from inference_local import load_model, get_reference_style, synthesize

def test_checkpoint(checkpoint_path, text, output_path, ref_s, model=None, diffusion_sampler=None, device=None):
    """Test a single checkpoint"""
    
    # Load model if not provided
    if model is None:
        model, diffusion_sampler, device = load_model(checkpoint_path)
        if model is None:
            return False
        ref_s = get_reference_style(model, device)
    
    # Generate audio
    wav = synthesize(text, model, ref_s, diffusion_sampler, device)
    
    if wav is not None:
        torchaudio.save(output_path, torch.from_numpy(wav).unsqueeze(0), 24000)
        return True
    return False

def main():
    """Compare all downloaded checkpoints"""
    
    print("="*70)
    print("🎯 Checkpoint Comparison Tool")
    print("="*70)
    
    # Your downloaded epochs
    checkpoint_dir = "Models/Cigdem_TTS"
    epochs_to_test = [4, 9, 14]
    
    # Test sentences (Turkish)
    test_sentences = [
        "Merhaba, ben Cigdem. Nasılsınız?",
        "Türkiye Cumhuriyeti vatandaşı mısınız?",
        "Bugün hava çok güzel.",
    ]
    
    # Or get custom input
    print("\nTest sentences:")
    for i, sent in enumerate(test_sentences, 1):
        print(f"  {i}. {sent}")
    
    custom = input("\nAdd custom text (or press Enter to use defaults): ").strip()
    if custom:
        test_sentences.append(custom)
    
    print("\n" + "-"*70)
    print("Testing epochs: 4, 9, 14")
    print("-"*70)
    
    # Test each epoch
    for epoch in epochs_to_test:
        checkpoint_name = f"epoch_2nd_{epoch:05d}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  Epoch {epoch}: {checkpoint_name} not found, skipping...")
            continue
        
        print(f"\n📦 Loading Epoch {epoch}...")
        model, diffusion_sampler, device = load_model(checkpoint_path)
        
        if model is None:
            continue
        
        # Get reference style
        ref_s = get_reference_style(model, device)
        
        # Test each sentence
        for i, text in enumerate(test_sentences, 1):
            output_file = f"compare_epoch{epoch:02d}_sent{i}.wav"
            print(f"  Generating: sentence {i}... ", end='')
            
            success = test_checkpoint(checkpoint_path, text, output_file, ref_s, 
                                     model, diffusion_sampler, device)
            
            if success:
                print(f"✅ {output_file}")
            else:
                print(f"❌ Failed")
        
        # Clean up GPU memory
        del model, diffusion_sampler
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  compare_epoch04_sent1.wav - Epoch 4, Sentence 1")
    print("  compare_epoch09_sent1.wav - Epoch 9, Sentence 1")
    print("  compare_epoch14_sent1.wav - Epoch 14, Sentence 1")
    print("  ... etc.")
    print("\n💡 Listen to each epoch's outputs to compare quality!")
    print("   Generally: Higher epoch = Better quality")

if __name__ == '__main__':
    main()
