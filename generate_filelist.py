import os

def generate_vctk_filelist_split_dirs(wav_dir, txt_dir, output_path, wav_ext=".wav"):
    """
    Generates a file list for FreeVC in the format: path_to_wav|text_of_audio
    using VCTK layout where audio and text are in separate root dirs.

    Args:
        wav_dir (str): Path to 'wav48' directory.
        txt_dir (str): Path to 'txt' directory.
        output_path (str): Where to save the output file list.
        wav_ext (str): Audio file extension (default: '.wav').
    """
    entries = []

    for speaker in sorted(os.listdir(wav_dir)):
        wav_speaker_dir = os.path.join(wav_dir, speaker)
        txt_speaker_dir = os.path.join(txt_dir, speaker)

        if not os.path.isdir(wav_speaker_dir) or not os.path.isdir(txt_speaker_dir):
            continue

        for file in sorted(os.listdir(wav_speaker_dir)):
            if file.endswith(wav_ext):
                wav_path = os.path.join(wav_speaker_dir, file)
                txt_path = os.path.join(txt_speaker_dir, file.replace(wav_ext, ".txt"))
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    entries.append(f"{wav_path}|{text}")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(entries))
    
    print(f"Saved {len(entries)} entries to {output_path}")

generate_vctk_filelist_split_dirs(
    wav_dir="/content/vietlish_accent_conversion/vctk16k",
    txt_dir="../drive/MyDrive/mini-vctk/txt",
    output_path="vctk_filelist.txt"
)
