#!/usr/bin/env python3
import os
import argparse
import torch
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm
import time

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder
from text import text_to_sequence
import commons

def get_text(text: str, hps) -> torch.LongTensor:
    """Turn raw text into model token IDs, optionally interspersed with blanks."""
    seq = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        seq = commons.intersperse(seq, 0)
    return torch.LongTensor(seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreeVC Text→Speech Inference")
    parser.add_argument("--hpfile",  type=str, required=True, help="path to config JSON")
    parser.add_argument("--ptfile",  type=str, required=True, help="path to model .pth")
    parser.add_argument("--txtpath", type=str, required=True,
                        help="path to text file (title|text|tgt_audio.wav per line)")
    parser.add_argument("--outdir",  type=str, default="outputs/groundtruth",
                        help="where to save the wavs")
    parser.add_argument("--use_timestamp", action="store_true",
                        help="prepend timestamp to filenames")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    # —— Load model —— 
    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda().eval()
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    # —— Speaker encoder —— 
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # —— Read lines —— 
    print(f"Reading lines from {args.txtpath} …")
    lines = []
    with open(args.txtpath, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw: continue
            title, text, tgt_wav = raw.split("|")
            lines.append((title, text, tgt_wav))

    # —— Inference loop —— 
    print("Synthesizing from text…")
    with torch.no_grad():
        for title, text, tgt_wav in tqdm(lines):
            # 1) Speaker embedding from tgt_wav
            wav, _ = librosa.load(tgt_wav, sr=hps.data.sampling_rate)
            wav = librosa.effects.trim(wav, top_db=20)[0]
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav = torch.from_numpy(wav).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav, 
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )                           

            # 2) Text → token IDs
            seq = get_text(text, hps)
            x_tst = seq.cuda().unsqueeze(0)                          # [1, L]
            x_len = torch.LongTensor([seq.size(0)]).cuda()           # [1]

            # 3) Inference (text path)
            #    net_g.infer picks the t_enc_p path when you use c_text=…
            if hps.model.use_spk:
                audio_tensor = net_g.infer(c_text=x_tst, c_text_lengths=x_len, g=g_tgt)
            else:
                audio_tensor = net_g.infer(c_text=x_tst, c_text_lengths=x_len, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()                                     # [T_out]

            # 4) Write out
            fname = f"{title}.wav"
            if args.use_timestamp:
                ts = time.strftime("%m-%d_%H-%M", time.localtime())
                fname = f"{ts}_{fname}"
            write(os.path.join(args.outdir, fname), hps.data.sampling_rate, audio)

    print(f"Done! Files written to {args.outdir}")
