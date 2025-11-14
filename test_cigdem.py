import torch
import torchaudio
import yaml
import numpy as np
from models import *
from Utils.PLBERT.util import load_plbert
from munch import Munch

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    params = checkpoint['net'] if 'net' in checkpoint else checkpoint
    
    # Load each model component
    for key in model:
        if key in params:
            print(f'  ✓ {key} loaded')
            try:
                model[key].load_state_dict(params[key], strict=False)
            except:
                print(f'  ⚠️ {key} failed to load, skipping...')
    
    return model

def compute_style(path):
    """Load and process reference audio to mel-spectrogram (Windows-compatible)"""
    wave, sr = torchaudio.load(path)
    
    # Convert to mono if stereo
    if wave.size(0) > 1:
        wave = torch.mean(wave, dim=0, keepdim=True)
    
    # Resample to 24kHz if needed (Windows-compatible method)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
        wave = resampler(wave)
    
    # Remove channel dimension for mel spectrogram
    wave = wave.squeeze(0)
    
    # Convert to mel-spectrogram (matching training parameters)
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mel_tensor = to_mel(wave)
    
    # Normalize (matching training preprocessing)
    mean, std = -4, 4
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    
    return mel_tensor.squeeze(0)  # Return [80, time]

def test_model(
    text="Merhaba, benim adım Çiğdem. Nasılsınız?",
    ref_audio_path="Data/my_voice/questions/question_001.wav",
    checkpoint_path="Models/Cigdem_TTS/checkpoints/epoch_2nd_00001.pth",
    output_path="output_cigdem.wav"
):
    print("🔄 Loading model...")
    
    # Load config
    config = yaml.safe_load(open("Configs/config_ft.yml"))
    
    # Load PLBERT
    print("📚 Loading PLBERT...")
    plbert = load_plbert('Utils/PLBERT/')
    
    # Load ASR and pitch extractor
    print("🎵 Loading ASR and pitch extractor...")
    from Utils.ASR.models import ASRCNN
    asr_config = yaml.safe_load(open('Utils/ASR/config.yml'))
    asr_model_path = 'Utils/ASR/epoch_00080.pth'
    
    text_aligner = ASRCNN(**asr_config['model_params'])
    asr_checkpoint = torch.load(asr_model_path, map_location='cpu', weights_only=False)
    # Handle different checkpoint formats
    if 'net' in asr_checkpoint:
        text_aligner.load_state_dict(asr_checkpoint['net'])
    elif 'model' in asr_checkpoint:
        text_aligner.load_state_dict(asr_checkpoint['model'])
    else:
        text_aligner.load_state_dict(asr_checkpoint)
    text_aligner.eval()
    text_aligner = text_aligner.cuda()
    
    from Utils.JDC.model import JDCNet
    pitch_extractor = JDCNet(num_class=1, seq_len=192)
    jdc_checkpoint = torch.load('Utils/JDC/bst.t7', map_location='cpu', weights_only=False)
    if 'net' in jdc_checkpoint:
        pitch_extractor.load_state_dict(jdc_checkpoint['net'])
    elif 'model' in jdc_checkpoint:
        pitch_extractor.load_state_dict(jdc_checkpoint['model'])
    else:
        pitch_extractor.load_state_dict(jdc_checkpoint)
    pitch_extractor.eval()
    pitch_extractor = pitch_extractor.cuda()
    
    # Build model
    print("🏗️ Building model...")
    from models import build_model
    
    # Recursively convert dict to Munch
    def to_munch(d):
        if isinstance(d, dict):
            return Munch({k: to_munch(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [to_munch(item) for item in d]
        else:
            return d
    
    model_params = to_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    # Load checkpoint
    print(f"💾 Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path)
    
    # Move all models to CUDA and set to eval
    _ = [model[key].cuda() for key in model]
    _ = [model[key].eval() for key in model]
    
    # Load reference audio
    print(f"🎤 Loading reference audio: {ref_audio_path}")
    ref_audio = compute_style(ref_audio_path)  # [80, time]
    ref_audio = ref_audio.unsqueeze(0).cuda()  # [1, 80, time]
    
    # Text to phonemes
    print(f"📝 Processing text: {text}")
    from text_utils import TextCleaner
    textcleaner = TextCleaner()
    
    # Convert text to token indexes
    tokens = textcleaner(text)
    tokens.insert(0, 0)  # Add start token
    tokens.append(0)  # Add end token
    tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
    
    print(f"🔢 Tokens: {tokens.shape}")
    
    # Generate speech
    print("🎵 Generating speech...")
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.size(-1)]).cuda()
        text_mask = length_to_mask(input_lengths).cuda()
        
        # Get text encoding
        t_en = model['text_encoder'](tokens, input_lengths, text_mask)
        
        # Get BERT embeddings
        bert_dur = model['bert'](tokens, attention_mask=(~text_mask).int())
        d_en = model['bert_encoder'](bert_dur).transpose(-1, -2)
        
        # Style encoding from reference
        s = model['style_encoder'](ref_audio.unsqueeze(1))
        
        # Predict duration
        d = model['predictor'].text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model['predictor'].lstm(d)
        duration = model['predictor'].duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        
        # Create alignment
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        
        # Encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).cuda())
        if config['model_params']['decoder']['type'] == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new
        
        # Predict F0 and energy
        F0_pred, N_pred = model['predictor'].F0Ntrain(en, s)
        
        # DEBUG: Check for NaN
        print(f"🔍 F0_pred - NaN: {torch.isnan(F0_pred).any().item()}, Min: {F0_pred.min().item():.4f}, Max: {F0_pred.max().item():.4f}")
        print(f"🔍 N_pred - NaN: {torch.isnan(N_pred).any().item()}, Min: {N_pred.min().item():.4f}, Max: {N_pred.max().item():.4f}")
        
        # ASR alignment
        asr = (t_en @ pred_aln_trg.unsqueeze(0).cuda())
        if config['model_params']['decoder']['type'] == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new
        
        print(f"🔍 asr - NaN: {torch.isnan(asr).any().item()}")
        print(f"🔍 s - NaN: {torch.isnan(s).any().item()}")
        
        # Generate waveform
        # decoder expects: (asr, F0_pred, N_pred, s)
        # where F0_pred and N_pred are [batch, time], s is [batch, style_dim]
        out = model['decoder'](asr, F0_pred, N_pred, s)
        
        print(f"🔍 decoder output - NaN: {torch.isnan(out).any().item()}, Min: {out.min().item():.4f}, Max: {out.max().item():.4f}")
    
    # Save output
    print(f"💾 Saving audio to: {output_path}")
    audio_out = out.squeeze().cpu()[..., :-50]
    if audio_out.dim() == 1:
        audio_out = audio_out.unsqueeze(0)
    torchaudio.save(output_path, audio_out, 24000)
    print("✅ Done!")
    
    return output_path

if __name__ == "__main__":
    # Test 1: Simple greeting
    print("\n" + "="*50)
    print("TEST 1: Simple Greeting")
    print("="*50)
    test_model(
        text="Merhaba, benim adım Çiğdem.",
        output_path="output_greeting.wav"
    )
    
    # Test 2: Question from your training data
    print("\n" + "="*50)
    print("TEST 2: Survey Question")
    print("="*50)
    test_model(
        text="Hükümetin asgari ücrete yaptığı artışa yönelik kanaatiniz nedir?",
        output_path="output_question.wav"
    )
    
    # Test 3: Custom text
    print("\n" + "="*50)
    print("TEST 3: Custom Text")
    print("="*50)
    custom_text = input("\n💬 Enter Turkish text to synthesize: ")
    if custom_text:
        test_model(
            text=custom_text,
            output_path="output_custom.wav"
        )
    else:
        print("⏭️ Skipping custom text test")
