"""
Inference script for StyleTTS2 Cigdem TTS
Generate speech from text using your trained model
"""

import torch
import torchaudio
import yaml
import numpy as np
from munch import Munch
import click

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# Text cleaner for Turkish text
textclenaer = TextCleaner()

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def inference(text, ref_s, model, diffusion_sampler, sampler):
    """
    Generate speech from text
    """
    text = text.strip()
    
    # Clean and prepare text
    ps = textclenaer(text)
    ps.insert(0, 0)
    ps.append(0)
    ps = torch.LongTensor(ps).unsqueeze(0).cuda()
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([ps.shape[-1]]).cuda()
        text_mask = length_to_mask(input_lengths).to('cuda')
        
        # Get text embeddings
        t_en = model.text_encoder(ps, input_lengths, text_mask)
        
        # Get style from reference
        bert_dur = model.bert(ps, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        
        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).cuda(),
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
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).cuda())
        
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        
        asr = (t_en @ pred_aln_trg.unsqueeze(0).cuda())
        
        # Generate mel spectrogram
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
        # Convert to waveform using diffusion
        out = diffusion_sampler(out.squeeze().unsqueeze(0).transpose(1, 2),
                              embedding=bert_dur).squeeze().cpu().numpy()
        
    return out

@click.command()
@click.option('--checkpoint', '-c', default='Models/Cigdem_TTS/epoch_2nd_00070.pth', 
              help='Path to checkpoint file')
@click.option('--config', default='Configs/config_ft.yml', 
              help='Path to config file')
@click.option('--text', '-t', default=None, 
              help='Text to synthesize (Turkish)')
@click.option('--output', '-o', default='output.wav', 
              help='Output audio file path')
@click.option('--ref_audio', '-r', default=None,
              help='Reference audio for style (optional, will use random from training data if not provided)')
def main(checkpoint, config, text, output, ref_audio):
    """
    Generate speech from text using trained Cigdem TTS model
    
    Example:
        python inference.py -t "Merhaba, nasılsınız?" -o test.wav
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load config
    config = yaml.safe_load(open(config))
    
    # Load models
    print('Loading models...')
    
    # Load ASR
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # Load F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # Load PLBERT
    PLBERT_dir = config.get('PLBERT_dir', False)
    plbert = load_plbert(PLBERT_dir)
    
    # Build model
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    
    # Load checkpoint
    print(f'Loading checkpoint: {checkpoint}')
    params_whole = torch.load(checkpoint, map_location='cpu')
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
                    name = k[7:] # remove `module.`
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
    
    # Style sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    # Get reference style
    if ref_audio:
        print(f'Using reference audio: {ref_audio}')
        # Load reference audio
        ref_wav, sr = torchaudio.load(ref_audio)
        if sr != 24000:
            ref_wav = torchaudio.functional.resample(ref_wav, sr, 24000)
        ref_wav = ref_wav.mean(0) if ref_wav.size(0) > 1 else ref_wav[0]
        
        # Get mel spectrogram
        ref_mel = mel_spectrogram(ref_wav.unsqueeze(0), 2048, 80, 24000, 1200, 300, 0, 8000)
        ref_mel = ref_mel.to(device)
        
        # Get reference style
        with torch.no_grad():
            ref_s = model.style_encoder(ref_mel.unsqueeze(1))
    else:
        print('Using random reference style from training data')
        # Use a random reference from training data
        train_list = config['data_params']['train_data']
        with open(train_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        import random
        ref_line = random.choice(lines).strip().split('|')[0]
        print(f'Reference: {ref_line}')
        
        ref_wav, sr = torchaudio.load(ref_line)
        if sr != 24000:
            ref_wav = torchaudio.functional.resample(ref_wav, sr, 24000)
        ref_wav = ref_wav.mean(0) if ref_wav.size(0) > 1 else ref_wav[0]
        
        ref_mel = mel_spectrogram(ref_wav.unsqueeze(0), 2048, 80, 24000, 1200, 300, 0, 8000)
        ref_mel = ref_mel.to(device)
        
        with torch.no_grad():
            ref_s = model.style_encoder(ref_mel.unsqueeze(1))
    
    # Interactive mode if no text provided
    if text is None:
        print('\n' + '='*60)
        print('🎤 Cigdem TTS - Interactive Mode')
        print('='*60)
        print('Enter Turkish text to synthesize (or "quit" to exit)')
        print('-'*60)
        
        counter = 1
        while True:
            text = input('\nText: ').strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print('Goodbye!')
                break
            
            if not text:
                continue
            
            try:
                print(f'Generating speech for: "{text}"')
                wav = inference(text, ref_s, model, diffusion_sampler, sampler)
                
                # Save audio
                output_file = f'output_{counter}.wav'
                torchaudio.save(output_file, torch.from_numpy(wav).unsqueeze(0), 24000)
                print(f'✅ Saved to: {output_file}')
                counter += 1
                
            except Exception as e:
                print(f'❌ Error: {e}')
                import traceback
                traceback.print_exc()
    else:
        # Single generation mode
        print(f'\nGenerating speech for: "{text}"')
        wav = inference(text, ref_s, model, diffusion_sampler, sampler)
        
        # Save audio
        torchaudio.save(output, torch.from_numpy(wav).unsqueeze(0), 24000)
        print(f'✅ Saved to: {output}')

if __name__ == '__main__':
    main()
