# Local Testing Setup Guide

## 📋 Prerequisites

Make sure you have everything installed:

```powershell
# Check if dependencies are installed
pip list | findstr "torch torchaudio"
```

If not installed, run:
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 📁 Setup Your Checkpoints

1. Put your downloaded epoch files in the correct location:

```
StyleTTS2/
  Models/
    Cigdem_TTS/
      epoch_2nd_00004.pth  ← Your epoch 4
      epoch_2nd_00009.pth  ← Your epoch 9
      epoch_2nd_00014.pth  ← Your epoch 14
```

2. Create the directory if it doesn't exist:

```powershell
mkdir Models\Cigdem_TTS
```

3. Copy your downloaded files there:

```powershell
copy path\to\downloaded\epoch_*.pth Models\Cigdem_TTS\
```

## 🎤 Testing Options

### Option 1: Interactive Testing (Best for trying different texts)

```powershell
python inference_local.py
```

This will:
- Show you all available checkpoints
- Let you select which one to test
- Interactive mode: type text and generate speech
- Auto-plays audio on Windows

### Option 2: Compare All Epochs (Best for quality comparison)

```powershell
python compare_checkpoints.py
```

This will:
- Test all 3 epochs (4, 9, 14) automatically
- Generate same sentences with all epochs
- Save files like: `compare_epoch04_sent1.wav`, `compare_epoch09_sent1.wav`, etc.
- You can then listen to compare quality

### Option 3: Quick Single Test

```powershell
# Test a specific checkpoint with specific text
python -c "from inference_local import *; import sys; model, diff, dev = load_model('Models/Cigdem_TTS/epoch_2nd_00014.pth'); ref = get_reference_style(model, dev); wav = synthesize('Merhaba dünya', model, ref, diff, dev); import torchaudio; torchaudio.save('test.wav', torch.from_numpy(wav).unsqueeze(0), 24000)"
```

## 🔍 Quick Test

Simplest test to see if everything works:

```powershell
cd C:\Users\Elxan\Desktop\StyleTTS2
python inference_local.py
```

Then:
1. Select checkpoint (or press Enter for latest)
2. Type: `Merhaba, nasılsınız?`
3. Wait for generation
4. Audio auto-plays and saves to `output_1.wav`

## 🎯 Expected Results

**Epoch 4:** Early training, may sound robotic
**Epoch 9:** Better, more natural  
**Epoch 14:** Should sound quite good
**Epoch 70+:** Best quality (if you continue training)

## ⚠️ Troubleshooting

**Error: "No module named 'models'"**
- Make sure you're in the StyleTTS2 directory
- Run: `cd C:\Users\Elxan\Desktop\StyleTTS2`

**Error: "Checkpoint not found"**
- Check file is in `Models/Cigdem_TTS/`
- Check filename matches: `epoch_2nd_00004.pth` (with zeros)

**CUDA out of memory:**
- The script uses CPU if CUDA fails
- For faster generation, close other GPU applications

**No audio plays:**
- Audio file is still saved as `output_X.wav`
- Open manually in Windows Media Player

## 💡 Tips

1. **Test epoch 14 first** (likely best quality of the 3)
2. **Use Turkish text** - model trained on Turkish
3. **Keep sentences reasonable length** (not too long)
4. **Compare outputs** to see improvement across epochs

## 📊 Compare Quality

To systematically compare:

```powershell
python compare_checkpoints.py
```

Then listen to the generated files in order:
- Epoch 4 → 9 → 14 (you'll hear improvement!)
