import os
import argparse
from tqdm import tqdm
from random import shuffle

def load_transcript(txt_root, speaker, fname):
    txt_path = os.path.join(txt_root, speaker, fname.replace('.wav', '.txt'))
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            return f.read().strip()
    else:
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="/content/vietlish_accent_conversion/filelists/train.txt")
    parser.add_argument("--val_list", type=str, default="/content/vietlish_accent_conversion/filelists/val.txt")
    parser.add_argument("--test_list", type=str, default="/content/vietlish_accent_conversion/filelists/test.txt")
    parser.add_argument("--source_dir", type=str, default="/content/drive/MyDrive/dataset (1)/datasets/vctk-corpus/VCTK-Corpus/VCTK-Corpus/vctk-22", help="path to .wav files")
    parser.add_argument("--txt_dir", type=str, default="/content/drive/MyDrive/dataset (1)/datasets/vctk-corpus/VCTK-Corpus/VCTK-Corpus/txt", help="path to .txt transcripts")
    args = parser.parse_args()
    
    train, val, test = [], [], []

    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dir = os.path.join(args.source_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue

        wavs = [f for f in os.listdir(spk_dir) if f.endswith(".wav")]
        shuffle(wavs)

        val += [(speaker, fname) for fname in wavs[:2]]
        test += [(speaker, fname) for fname in wavs[-5:]]
        train += [(speaker, fname) for fname in wavs[2:-5]]

    shuffle(train)
    shuffle(val)
    shuffle(test)

    def write_list(file_path, data):
        print(f"Writing {file_path}")
        with open(file_path, "w") as f:
            for speaker, fname in tqdm(data):
                wavpath = os.path.join("DUMMY", speaker, fname)
                transcript = load_transcript(args.txt_dir, speaker, fname)
                if transcript:  # Only write if transcript exists
                    line = f"{wavpath}|{speaker}|{transcript}"
                    f.write(line + "\n")

    write_list(args.train_list, train)
    write_list(args.val_list, val)
    write_list(args.test_list, test)
