import os
import wave

def check_audio_files(train_list_path, root_path):
    """Check all audio files listed in the training list for corruption."""
    
    corrupted_files = []
    missing_files = []
    valid_files = []
    
    with open(train_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Checking {len(lines)} audio files...")
    print("-" * 60)
    
    for i, line in enumerate(lines, 1):
        try:
            parts = line.strip().split('|')
            if len(parts) < 2:
                print(f"Line {i}: Invalid format - {line.strip()}")
                continue
                
            audio_path = parts[0]
            full_path = os.path.join(root_path, audio_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                missing_files.append((i, audio_path))
                print(f"X Line {i}: MISSING - {audio_path}")
                continue
            
            # Check file size
            file_size = os.path.getsize(full_path)
            if file_size == 0:
                corrupted_files.append((i, audio_path, "Empty file (0 bytes)"))
                print(f"X Line {i}: CORRUPTED - {audio_path} - Empty file")
                continue
            
            # Try to open the audio file
            try:
                with wave.open(full_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0
                valid_files.append((i, audio_path, frames, rate))
                print(f"OK Line {i}: OK - {audio_path} ({duration:.2f}s, {rate} Hz, {file_size} bytes)")
            except Exception as e:
                corrupted_files.append((i, audio_path, str(e)))
                print(f"X Line {i}: CORRUPTED - {audio_path}")
                print(f"  Error: {e}")
                print(f"  File size: {file_size} bytes")
                
        except Exception as e:
            print(f"X Line {i}: ERROR processing line - {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files checked: {len(lines)}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if corrupted_files:
        print("\n" + "-" * 60)
        print("CORRUPTED FILES:")
        print("-" * 60)
        for line_num, path, error in corrupted_files:
            print(f"Line {line_num}: {path}")
            print(f"  Error: {error}")
    
    if missing_files:
        print("\n" + "-" * 60)
        print("MISSING FILES:")
        print("-" * 60)
        for line_num, path in missing_files:
            print(f"Line {line_num}: {path}")
    
    return corrupted_files, missing_files, valid_files

if __name__ == "__main__":
    train_list = r"Data\my_train_list.txt"
    root_path = r"."
    
    print("Audio File Integrity Check")
    print("=" * 60)
    print(f"Train list: {train_list}")
    print(f"Root path: {root_path}")
    print("=" * 60 + "\n")
    
    corrupted, missing, valid = check_audio_files(train_list, root_path)
    
    if corrupted or missing:
        print("\n⚠️  Found issues! Please fix the corrupted/missing files before training.")
    else:
        print("\n✓ All audio files are valid!")
