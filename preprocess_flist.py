import os
import argparse
from tqdm import tqdm
from random import shuffle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="path to source dir")
    args = parser.parse_args()

    # Automatically determine which set we are processing
    source_type = os.path.basename(args.source_dir.rstrip("/"))
    output_dir = os.path.join("filelists", source_type)
    os.makedirs(output_dir, exist_ok=True)

    # Filelist paths
    train_list = os.path.join(output_dir, "train.txt")
    val_list = os.path.join(output_dir, "val.txt")
    test_list = os.path.join(output_dir, "test.txt")

    train = []
    val = []
    test = []

    for speaker in tqdm(os.listdir(args.source_dir)):
        speaker_path = os.path.join(args.source_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        wavs = os.listdir(speaker_path)
        wavs = [f for f in wavs if f.endswith(".wav")]
        shuffle(wavs)

        val += [os.path.join(speaker, f) for f in wavs[:2]]
        test += [os.path.join(speaker, f) for f in wavs[-10:]]
        train += [os.path.join(speaker, f) for f in wavs[2:-10]]

    # Final shuffle
    shuffle(train)
    shuffle(val)
    shuffle(test)

    print("Writing", train_list)
    with open(train_list, "w") as f:
        for fname in tqdm(train):
            f.write(os.path.join("DUMMY", fname) + "\n")

    print("Writing", val_list)
    with open(val_list, "w") as f:
        for fname in tqdm(val):
            f.write(os.path.join("DUMMY", fname) + "\n")

    print("Writing", test_list)
    with open(test_list, "w") as f:
        for fname in tqdm(test):
            f.write(os.path.join("DUMMY", fname) + "\n")
