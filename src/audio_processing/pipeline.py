import os
import webrtcvad
import pickle

from .resample import resample_to_pcm16
from .vad import read_wave, frame_generator, vad_collector
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='webrtcvad')


def get_speech_windows(segments, max_gap_ms=700):
    """
    Merges consecutive speech segments into larger speech windows,
    allowing short gaps (≤ max_gap_ms) between them.

    :param segments: list of tuples
        List of (start_ms, end_ms) speech segments.

    :param max_gap_ms: int
        Maximum gap (in milliseconds) allowed between segments to be merged.

    :return: list of dict
        Merged speech windows, each as a dict with "start" and "end" keys (in ms).
    """
    if not segments:
        return []

    merged = []
    current_start, current_end = segments[0]

    for seg_start, seg_end in segments[1:]:
        if seg_start - current_end <= max_gap_ms:
            # Merge with current window
            current_end = seg_end
        else:
            # Save current window as dict and start new
            merged.append({"start": current_start, "end": current_end})
            current_start, current_end = seg_start, seg_end

    # Append final window
    merged.append({"start": current_start, "end": current_end})
    return merged


def process_audio_file(path, tmp_pcm="tmp_16k_mono.wav"):
    """
    Applies VAD to a single audio file and returns merged speech regions.

    :param path: str
        Path to the input audio file (.wav or any ffmpeg-compatible format).

    :param tmp_pcm: str
        Temporary filename for the 16kHz mono PCM-converted version.

    :return: list of dict
        Speech windows as a list of {"start": ms, "end": ms} dictionaries.
    """
    # Resample input to 16kHz mono PCM
    resample_to_pcm16(path, tmp_pcm)
    audio, sr = read_wave(tmp_pcm)
    vad = webrtcvad.Vad(2)
    frames = list(frame_generator(30, audio, sr))
    segments = list(vad_collector(sr, 30, 700, vad, frames))
    os.remove(tmp_pcm)

    # Convert merged windows to list of dicts
    return get_speech_windows(segments, max_gap_ms=700)


def run_audio_processing(pywork_output_dir, audio_dir):
    """
    Processes all .wav files in a directory to detect speech regions.
    Saves the result as a pickle mapping filename → speech windows.

    :param pywork_output_dir: str
        Output directory where 'speech_windows.pkl' will be saved.

    :param audio_dir: str
        Directory containing the input .wav audio files.

    :return: None
        Outputs a 'speech_windows.pkl' file containing:
            { filename: [ {start: ms, end: ms}, ... ] }
    """
    output_path = os.path.join(pywork_output_dir, 'speech_windows.pkl')

    results = {}
    for fname in os.listdir(audio_dir):
        print("Processing {}".format(fname))
        if fname.lower().endswith(".wav"):
            path = os.path.join(audio_dir, fname)
            print(f"Processing {fname}...")
            windows = process_audio_file(path)
            # windows already a list of dicts
            results[fname] = windows

    # Save the mapping: filename -> [ {start, end}, ... ]
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved speech windows to {output_path}")

