import os

from .compute_cosine_similarity import compute_cosine_similarity
from .utils import load_and_group_av_cosine
from .compute_probability import compute_probability
from .plot_probabilities import plot_probs


def run_speaker_assignment(work_dir, audio_dir, temp_dir, crop_dir, graphs_dir, weights_path, batch_size=20,
                           temperature=0.5, smooth_size=10, ema_alpha=0.1, frame_step_ms=40):

    """
    Runs the full speaker assignment pipeline, from audio-video similarity computation to probability plotting.

    Steps:
      1. Computes cosine similarity between audio segments and face tracks using SyncNet.
      2. Groups cosine similarity results by audio file and track.
      3. Computes softmax-based speaker probabilities for each audio segment across tracks.
      4. Applies smoothing and visualizes speaker probability timelines.

    :param work_dir: str
        Directory for reading and saving intermediate and final data (e.g., pickles, logs).

    :param audio_dir: str
        Directory containing input WAV audio files.

    :param temp_dir: str
        Temporary directory.

    :param crop_dir: str
        Directory containing cropped face track videos (e.g., '00001.avi').

    :param graphs_dir: str
        Output directory where probability plots (.png) will be saved.

    :param weights_path: str
        Path to the SyncNet model weights file (.pth).

    :param batch_size: int, optional
        Batch size used during inference with the SyncNet model (default: 20).

    :param temperature: float, optional
        Softmax temperature for probability computation (default: 0.5).

    :param smooth_size: int, optional
        Window size for uniform smoothing of softmax probabilities (default: 10).

    :param ema_alpha: float, optional
        Smoothing coefficient for EMA smoothing of probabilities (default: 0.1).

    :param frame_step_ms: int, optional
        Time step between audio frames in milliseconds (default: 40).

    :return: None
        Results are saved to disk; no value is returned.
    """

    # Get Cosine Similarity
    compute_cosine_similarity(work_dir, audio_dir, temp_dir, crop_dir, weights_path,
                              os.path.join(work_dir, 'speech_windows.pkl'),  os.path.join(work_dir, 'tracks.pkl'),
                              batch_size)

    # Group segments together by track
    load_and_group_av_cosine(os.path.join(work_dir, 'av_cosine.pkl'), os.path.join(work_dir, 'av_cosine_grouped.pkl'))

    # Compute Probabilities
    compute_probability(os.path.join(work_dir, 'av_cosine_grouped.pkl'), os.path.join(work_dir, 'probabilities.pkl'),
                        temperature, smooth_size, ema_alpha, frame_step_ms)

    # Plot Probabilities
    plot_probs(os.path.join(work_dir, 'probabilities.pkl'), graphs_dir)



