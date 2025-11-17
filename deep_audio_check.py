"""
Deep audio file checker - simulates exactly how meldataset.py loads files
This will catch issues that simple wave module checks miss
"""
import os
import numpy as np
import librosa

def check_with_librosa(train_list_path, root_path):
    """Check all audio files using librosa (like the training code does)"""
    
    corrupted_files = []
    missing_files = []
    valid_files = []
    
    with open(train_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Checking {len(lines)} audio files with librosa...")
    print("-" * 60)
    
    for i, line in enumerate(lines, 1):
        parts = line.strip().split('|')
        if len(parts) < 2:
            print(f"Line {i}: Invalid format")
            continue
            
        audio_path = parts[0]
        full_path = os.path.join(root_path, audio_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            missing_files.append((i, audio_path))
            print(f"X Line {i}: MISSING - {audio_path}")
            continue
        
        # Try to load exactly as the training code does
        try:
            # This is how meldataset.py loads files
            import soundfile as sf
            wave, sr = sf.read(full_path)
            
            # Check for stereo and convert
            if len(wave.shape) > 1 and wave.shape[-1] == 2:
                wave = wave[:, 0].squeeze()
            
            # Check if resampling is needed
            if sr != 24000:
                print(f"  Note: File has {sr} Hz, will be resampled to 24000 Hz")
                wave_resampled = librosa.resample(wave, orig_sr=sr, target_sr=24000)
                wave = wave_resampled
            
            # Add padding (like training code does)
            wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
            
            duration = len(wave) / 24000
            valid_files.append((i, audio_path))
            print(f"OK Line {i}: {audio_path} ({duration:.2f}s, {sr} Hz -> 24000 Hz)")
            
        except Exception as e:
            corrupted_files.append((i, audio_path, str(e)))
            print(f"X Line {i}: FAILED - {audio_path}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(lines)}")
    print(f"Valid: {len(valid_files)}")
    print(f"Corrupted: {len(corrupted_files)}")
    print(f"Missing: {len(missing_files)}")
    
    if corrupted_files:
        print("\n" + "-" * 60)
        print("CORRUPTED/PROBLEMATIC FILES:")
        print("-" * 60)
        for line_num, path, error in corrupted_files:
            print(f"Line {line_num}: {path}")
            print(f"  {error}")
    
    return corrupted_files, missing_files, valid_files

if __name__ == "__main__":
    # Check if soundfile is available
    try:
        import soundfile as sf
        print("soundfile library: FOUND")
    except ImportError:
        print("ERROR: soundfile not installed!")
        print("Install with: pip install soundfile")
        exit(1)
    
    try:
        import librosa
        print("librosa library: FOUND")
    except ImportError:
        print("ERROR: librosa not installed!")
        print("Install with: pip install librosa")
        exit(1)
    
    print("\n" + "=" * 60)
    print("Deep Audio File Check (using soundfile + librosa)")
    print("=" * 60 + "\n")
    
    train_list = r"Data\my_train_list.txt"
    root_path = r"."
    
    corrupted, missing, valid = check_with_librosa(train_list, root_path)
    
    if corrupted or missing:
        print("\n⚠️  ISSUES FOUND - These files will cause training to fail!")
    else:
        print("\n✓ All files passed deep check!")
