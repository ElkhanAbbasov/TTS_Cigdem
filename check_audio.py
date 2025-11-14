"""Check if audio files contain actual sound or just silence"""
import torchaudio
import torch
import numpy as np

def analyze_audio(filepath):
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*60}")
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(filepath)
        
        print(f"✅ File loaded successfully")
        print(f"📊 Sample rate: {sample_rate} Hz")
        print(f"📏 Shape: {waveform.shape}")
        print(f"⏱️  Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
        
        # Convert to numpy for analysis
        audio_np = waveform.numpy().flatten()
        
        # Calculate statistics
        max_val = np.abs(audio_np).max()
        mean_val = np.abs(audio_np).mean()
        rms = np.sqrt(np.mean(audio_np**2))
        non_zero = np.count_nonzero(audio_np)
        
        print(f"\n📈 Audio Statistics:")
        print(f"  • Max amplitude: {max_val:.6f}")
        print(f"  • Mean amplitude: {mean_val:.6f}")
        print(f"  • RMS: {rms:.6f}")
        print(f"  • Non-zero samples: {non_zero}/{len(audio_np)} ({100*non_zero/len(audio_np):.1f}%)")
        
        # Diagnosis
        if max_val < 0.0001:
            print(f"\n❌ SILENT: Audio is essentially silent (max < 0.0001)")
            print(f"   Possible causes:")
            print(f"   - Decoder not properly trained")
            print(f"   - F0/pitch prediction is zeros")
            print(f"   - Model not learning from fine-tuning data")
        elif max_val < 0.01:
            print(f"\n⚠️  VERY QUIET: Audio is extremely quiet (max < 0.01)")
            print(f"   May need normalization or more training epochs")
        else:
            print(f"\n✅ CONTAINS SOUND: Audio has significant amplitude")
            print(f"   Should be audible when played")
        
        # Check for NaN or Inf
        has_nan = np.isnan(audio_np).any()
        has_inf = np.isinf(audio_np).any()
        if has_nan or has_inf:
            print(f"\n❌ WARNING: Audio contains NaN={has_nan}, Inf={has_inf}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Check all output files
files = [
    "output_greeting.wav",
    "output_question.wav", 
    "output_custom.wav"
]

for file in files:
    analyze_audio(file)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print("If all files show SILENT, the model needs more training epochs")
print("or there's an issue with the decoder/F0 prediction.")
print(f"{'='*60}\n")
