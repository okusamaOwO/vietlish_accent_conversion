import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm

def process(wav_name):
    speaker = wav_name[:4]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and wav_name.endswith('.wav'):
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)

        wav, sr = librosa.load(wav_path, sr=None)
        wav, _ = librosa.effects.trim(wav, top_db=20)

        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak

        wav_resampled = librosa.resample(wav, orig_sr=sr, target_sr=args.sr)

        save_name = wav_name.replace("_mic2.flac", ".wav")
        save_path = os.path.join(args.out_dir, speaker, save_name)

        wavfile.write(
            save_path,
            args.sr,
            (wav_resampled * np.iinfo(np.int16).max).astype(np.int16)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=22050, help="Target sampling rate (Hz)")
    parser.add_argument("--in_dir", type=str, default="/content/drive/MyDrive/dataset (1)/datasets/vctk-corpus/VCTK-Corpus/VCTK-Corpus/wav48", help="path to source dir")
    parser.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/dataset/vctk-22", help="Path to output dataset")
    args = parser.parse_args()

    pool = Pool(processes=max(1, cpu_count() - 2))

    all_speakers = sorted(
        [d for d in os.listdir(args.in_dir) if d >= "p241"]
    )
    for speaker in all_speakers:
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            wav_list = os.listdir(spk_dir)
            for _ in tqdm(pool.imap_unordered(process, wav_list), total=len(wav_list), desc=f"Processing {speaker}"):
                pass

    pool.close()
    pool.join()
