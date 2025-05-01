import os

def generate_vctk_filelist(vctk_root, output_path, wav_ext=".wav"):
    """
    Generates a file list in the format: path_to_wav|text_of_audio
    Assumes VCTK structure: {vctk_root}/{speaker}/{filename}.wav and .txt
    
    Parameters:
        vctk_root (str): Root directory of VCTK dataset.
        output_path (str): Path to write the output file list.
        wav_ext (str): Audio file extension (default: '.wav').
    """
    entries = []

    for speaker in sorted(os.listdir(vctk_root)):
        speaker_dir = os.path.join(vctk_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for file in sorted(os.listdir(speaker_dir)):
            if file.endswith(wav_ext):
                wav_path = os.path.join(speaker_dir, file)
                txt_path = wav_path.replace(wav_ext, ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    entries.append(f"{wav_path}|{text}")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(entries))

    print(f"File list saved to {output_path} with {len(entries)} entries.")

generate_vctk_filelist(
    vctk_root="/home/Datasets/lijingyi/data/vctk/wav48_silence_trimmed/",
    output_path="vctk_filelist.txt"
)
