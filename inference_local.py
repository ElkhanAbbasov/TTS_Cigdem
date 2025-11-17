"""
Local inference script for Windows PC
Test your trained Cigdem TTS model locally

Requirements:
- Python 3.8+
- CUDA (optional, but recommended for speed)
- All dependencies from requirements.txt

Usage:
    python inference_local.py
"""

import torch
import torchaudio
import yaml
import numpy as np
from munch import Munch
import os
import sys

# Check if we have required modules
try:
    from models import *
    from utils import *
    from text_utils import TextCleaner
    from Utils.PLBERT.util import load_plbert
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure you're running this from the StyleTTS2 directory!")
    sys.exit(1)

# Text cleaner for Turkish text
textcleaner = TextCleaner()

# Mel spectrogram transform
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def load_model(checkpoint_path, config_path='Configs/config_ft.yml'):
    """Load the trained model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load config
    config = yaml.safe_load(open(config_path))
    
    # Load models
    print('Loading models...')
    
    # Load ASR
    ASR_config = config.get('ASR_config', 'Utils/ASR/config.yml')
    ASR_path = config.get('ASR_path', 'Utils/ASR/epoch_00080.pth')
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # Load F0 model
    F0_path = config.get('F0_path', 'Utils/JDC/bst.t7')
    pitch_extractor = load_F0_models(F0_path)
    
    # Load PLBERT
    PLBERT_dir = config.get('PLBERT_dir', 'Utils/PLBERT/')
    plbert = load_plbert(PLBERT_dir)
    
    # Build model
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    
    # Load checkpoint
    print(f'Loading checkpoint: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None, None, None
        
    params_whole = torch.load(checkpoint_path, map_location='cpu')
    params = params_whole['net']
    
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    
    _ = [model[key].eval() for key in model]
    
    print('✅ Models loaded successfully!')
    
    # Setup diffusion sampler
    diffusion_sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    return model, diffusion_sampler, device

def get_reference_style(model, device, ref_audio_path=None):
    """Get reference style from audio"""
    
    if ref_audio_path and os.path.exists(ref_audio_path):
        print(f'Using reference audio: {ref_audio_path}')
        ref_wav, sr = torchaudio.load(ref_audio_path)
    else:
        # Use a random reference from training data
        train_list = 'Data/my_train_list.txt'
        if os.path.exists(train_list):
            with open(train_list, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            import random
            ref_line = random.choice(lines).strip().split('|')[0]
            print(f'Using training reference: {ref_line}')
            ref_wav, sr = torchaudio.load(ref_line)
        else:
            print("ERROR: No reference audio found!")
            return None
    
    if sr != 24000:
        ref_wav = torchaudio.functional.resample(ref_wav, sr, 24000)
    ref_wav = ref_wav.mean(0) if ref_wav.size(0) > 1 else ref_wav[0]
    
    # Get mel spectrogram
    ref_mel = to_mel(ref_wav)
    ref_mel = ref_mel.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get both style and predictor encodings (need 256 dims total)
        ref_ss = model.style_encoder(ref_mel.unsqueeze(1))  # 128 dims
        ref_sp = model.predictor_encoder(ref_mel.unsqueeze(1))  # 128 dims
        ref_s = torch.cat([ref_ss, ref_sp], dim=1)  # 256 dims total
    
    return ref_s

def synthesize(text, model, ref_s, diffusion_sampler, device):
    """Generate speech from text"""
    
    text = text.strip()
    if not text:
        return None
    
    # Clean and prepare text
    ps = textcleaner(text)
    if not ps:
        print(f"Warning: Text cleaning resulted in empty sequence for: {text}")
        return None
        
    ps.insert(0, 0)
    ps.append(0)
    ps = torch.LongTensor(ps).unsqueeze(0).to(device)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([ps.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        
        # Get text embeddings
        t_en = model.text_encoder(ps, input_lengths, text_mask)
        
        # Get style
        bert_dur = model.bert(ps, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        
        # Style sampler
        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )
        
        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        features=ref_s,
                        num_steps=10).squeeze(1)
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        
        # Get duration
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        
        # Encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        
        # Generate waveform directly with decoder (HiFiGAN)
        out = model.decoder(asr, F0_pred, N_pred, s)
        
        # Convert to numpy and trim
        out = out.squeeze().cpu().numpy()[..., :-50]  # Trim padding
    
    return out

def main():
    """Interactive testing on local PC"""
    
    print("="*70)
    print("🎤 Cigdem TTS - Local Testing")
    print("="*70)
    
    # List available checkpoints
    checkpoint_dir = "Models/Cigdem_TTS"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            print("\n📁 Available checkpoints:")
            for i, ckpt in enumerate(sorted(checkpoints), 1):
                size = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024*1024)
                print(f"  {i}. {ckpt} ({size:.1f} MB)")
        else:
            print("\n⚠️ No checkpoints found in Models/Cigdem_TTS/")
            print("Please copy your downloaded epoch files here.")
            return
    else:
        print(f"\n❌ Directory not found: {checkpoint_dir}")
        print("Please create it and copy your epoch files there.")
        return
    
    # Select checkpoint
    print("\n" + "-"*70)
    checkpoint_num = input("Select checkpoint number (or press Enter for latest): ").strip()
    
    if checkpoint_num:
        try:
            checkpoint_name = sorted(checkpoints)[int(checkpoint_num)-1]
        except:
            print("Invalid selection, using latest...")
            checkpoint_name = sorted(checkpoints)[-1]
    else:
        checkpoint_name = sorted(checkpoints)[-1]
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Using: {checkpoint_name}")
    
    # Load model
    print("\n" + "-"*70)
    model, diffusion_sampler, device = load_model(checkpoint_path)
    if model is None:
        return
    
    # Get reference style
    ref_s = get_reference_style(model, device)
    if ref_s is None:
        return
    
    # Interactive loop
    print("\n" + "="*70)
    print("✅ Ready! Type Turkish text to generate speech")
    print("Commands: 'quit' to exit, 'change' to switch checkpoint")
    print("="*70)
    
    counter = 1
    while True:
        print(f"\n[{counter}]")
        text = input("Text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if text.lower() == 'change':
            print("\n🔄 Restart the script to change checkpoint")
            continue
        
        if not text:
            continue
        
        try:
            print(f"🎤 Generating: '{text}'...")
            wav = synthesize(text, model, ref_s, diffusion_sampler, device)
            
            if wav is not None:
                # Save audio
                output_file = f'output_{counter}.wav'
                torchaudio.save(output_file, torch.from_numpy(wav).unsqueeze(0), 24000)
                print(f"✅ Saved: {output_file}")
                
                # Try to play on Windows
                try:
                    import winsound
                    print("🔊 Playing audio...")
                    winsound.PlaySound(output_file, winsound.SND_FILENAME)
                except:
                    print("(Auto-play not available, open the file manually)")
                
                counter += 1
            else:
                print("❌ Failed to generate audio")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
