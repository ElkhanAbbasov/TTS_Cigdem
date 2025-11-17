import wave
import os

# Read the training list
with open('Data/my_train_list.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Checking {len(lines)} audio files...\n")

corrupted_files = []
missing_files = []

for line in lines:
    filepath = line.strip().split('|')[0]
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"✗ {filepath}: FILE NOT FOUND")
        missing_files.append(filepath)
        continue
    
    # Check file size
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        print(f"✗ {filepath}: EMPTY FILE (0 bytes)")
        corrupted_files.append(filepath)
        continue
    
    # Try to open with wave module
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            print(f"✓ {filepath}: {file_size} bytes, {rate} Hz, {duration:.2f}s")
    except Exception as e:
        print(f"✗ {filepath}: ERROR - {str(e)}")
        corrupted_files.append(filepath)

print(f"\n{'='*60}")
if missing_files:
    print(f"Missing {len(missing_files)} file(s):")
    for f in missing_files:
        print(f"  - {f}")
if corrupted_files:
    print(f"Corrupted {len(corrupted_files)} file(s):")
    for f in corrupted_files:
        print(f"  - {f}")
if not missing_files and not corrupted_files:
    print("All files are OK!")
