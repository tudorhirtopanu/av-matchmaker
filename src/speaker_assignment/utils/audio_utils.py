import os
import subprocess

import torch
import math
import numpy as np
import python_speech_features
from scipy.io import wavfile


def ensure_audio_format(src_wav, tgt_wav):
    """
    Converts an input audio file to mono 16kHz WAV format using ffmpeg.

    :param src_wav: str
        Path to the source WAV file.
    :param tgt_wav: str
        Path to the target WAV file to be saved in mono 16kHz format.

    :return: tuple
        A tuple (sr, audio) where:
            - sr (int): Sample rate (should be 16000).
            - audio (np.ndarray): The loaded audio data as a float array.
    """

    os.makedirs(os.path.dirname(tgt_wav), exist_ok=True)

    # convert to mono and 16khz
    cmd = (
        f'ffmpeg -y -i "{src_wav}" '
        f'-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "{tgt_wav}"'
    )
    subprocess.call(cmd, shell=True)

    # load converted full audio
    sr, audio = wavfile.read(tgt_wav)
    audio = audio.astype(float)
    assert sr == 16000, f"Unexpected sample rate: {sr}"

    return sr, audio


def extract_audio_segments(segment_map, audio_dir, temp_dir):
    """
    Extracts MFCC features from audio segments defined in the segment map.

    :param segment_map: dict
        A dictionary mapping audio filenames to a list of segments.
        Each segment should be a dict with 'start' and 'end' times in milliseconds.

    :param audio_dir: str
        Directory containing the original WAV files.

    :param temp_dir: str
        Directory to save the preprocessed (converted) audio files.

    :return: list
        A list of dictionaries, each containing:
            - 'fname': str, original filename
            - 'seg_idx': int, index of the segment
            - 'cct': torch.Tensor, MFCC features as a 4D tensor [1×1×C×T]
            - 'num_samp': int, number of 40ms windows
            - 'start_ms': float, segment start time in milliseconds
            - 'end_ms': float, segment end time in milliseconds
    """

    audio_data = []
    os.makedirs(temp_dir, exist_ok=True)

    for fname, seg_list in segment_map.items():
        if not fname.lower().endswith('.wav'):
            continue

        src_wav = os.path.join(audio_dir, fname)
        tgt_wav = os.path.join(temp_dir, fname)

        if not os.path.exists(src_wav):
            print(f"Missing source audio {src_wav}, skipping.")
            continue

        sr, audio = ensure_audio_format(src_wav, tgt_wav)

        # Slice & MFCC per segment
        for seg_idx, seg in enumerate(seg_list):

            # ms -> sample indices
            start_samp = int(seg['start'] * sr / 1000)
            end_samp = int(seg['end'] * sr / 1000)
            seg_audio = audio[start_samp:end_samp]

            if len(seg_audio) < 1:
                continue

            # compute MFCC: [num_frames × num_coeffs]
            mfcc_feats = python_speech_features.mfcc(seg_audio, sr)

            # transpose -> [num_coeffs × num_frames]
            mfcc_arr = np.stack(list(zip(*mfcc_feats)), axis=0)  # [C × T]

            # to tensor [1×1×C×T]
            cct = torch.from_numpy(mfcc_arr[None, None, :, :].astype(float)).float()

            num_samp = math.floor(len(seg_audio) / 640)  # number of 40ms windows

            audio_data.append({
                'fname': fname,
                'seg_idx': seg_idx,
                'cct': cct,
                'num_samp': num_samp,
                'start_ms': seg['start'],
                'end_ms': seg['end'],
            })

        print(f"Preprocessed {fname}: {len(seg_list)} segment(s)")

    return audio_data
