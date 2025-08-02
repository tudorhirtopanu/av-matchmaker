import subprocess
import sys


def resample_to_pcm16(input_path, output_path):
    """
    Converts an audio file to 16-bit PCM mono at 16 kHz using ffmpeg.

    :param input_path: str
        Path to the input audio file (any format supported by ffmpeg).

    :param output_path: str
        Path where the converted .wav file will be saved.

    :return: None
        The converted file is written to disk. Exits the program if ffmpeg fails.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-acodec", "pcm_s16le",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed: {e}")
