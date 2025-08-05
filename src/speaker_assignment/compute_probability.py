import pickle
import os
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from collections import defaultdict


def stable_softmax(logits, temp=1.0):
    """
    Computes a numerically stable softmax over the last dimension of a 2D tensor.

    :param logits: torch.Tensor
        A tensor of shape [T, N], where T is the number of time steps and N is the number of classes (e.g., tracks).

    :param temp: float
        Temperature scaling parameter. Higher values produce a softer distribution.

    :return: torch.Tensor
        Softmax probabilities of shape [T, N].
    """
    x = logits / temp
    x = x - x.max(dim=1, keepdim=True).values
    exp = torch.exp(x)
    return exp / exp.sum(dim=1, keepdim=True)


def ema_smoothing(probs, alpha):
    """
    Applies Exponential Moving Average (EMA) smoothing over time for each track's probabilities.

    :param probs: np.ndarray
        Array of shape [T, N] containing per-window softmax probabilities over N tracks.

    :param alpha: float
        Smoothing factor (0 < alpha < 1). Higher values weight recent frames more heavily.

    :return: np.ndarray
        Smoothed probabilities of shape [T, N].
    """
    smoothed = probs.copy()
    for t in range(1, probs.shape[0]):
        smoothed[t] = alpha * probs[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def compute_probability(pkl_input_path, pkl_output_path, temperature, smooth_size, ema_alpha, frame_step_ms):
    """
    Computes per-frame softmax probabilities over speakers (face tracks) for each audio segment.

    - Loads grouped cosine similarity data from a pickle file.
    - For each segment, applies softmax over track-wise cosine values.
    - Applies both uniform and EMA smoothing.
    - Stores time-aligned probability outputs to a new pickle file.

    :param pkl_input_path: str
        Path to input pickle file containing grouped cosine similarity data.

    :param pkl_output_path: str
        Path to output pickle file for saving computed probabilities.

    :param temperature: float
        Temperature value for the softmax function. Controls output sharpness.

    :param smooth_size: int
        Window size for uniform (box) smoothing.

    :param ema_alpha: float
        Smoothing coefficient for exponential moving average (EMA) smoothing.

    :param frame_step_ms: int
        Step size in milliseconds between audio frames (e.g., 40 ms per frame).

    :return: dict
        Nested dictionary mapping:
            audio_file → segment_idx → {
                'track_ids': list of int,
                'per_window_probs': np.ndarray [T × N],
                'smoothed_uniform': np.ndarray [T × N],
                'smoothed_ema': np.ndarray [T × N],
                'time_s': np.ndarray [T],
                'start_ms': float,
                'end_ms': float,
                'num_windows': int
            }
    """
    if not os.path.isfile(pkl_input_path):
        raise FileNotFoundError(f"Pickle not found: {pkl_input_path}")

    with open(pkl_input_path, "rb") as f:
        grouped = pickle.load(f)  # list of dicts with audio_file, track_id, segments, metadata

    # Build mapping: (audio_file, segment_idx) -> list of (track_id, segment_dict)
    #
    # This groups all speaker candidates (tracks) for each specific audio segment.
    # It allows us to compare multiple face tracks (e.g., track 0 and 1) for the same segment
    # of a given audio file, so we can compute per-window softmax probabilities across speakers.
    #
    # Example structure:
    # {
    #   ("host.wav", 0): [ (0, segment_dict_for_track0), (1, segment_dict_for_track1) ],
    #   ("host.wav", 1): [ (0, ...), (1, ...) ]
    # }
    per_segment = defaultdict(list)
    for entry in grouped:
        audio_file = entry["audio_file"]
        track_id = entry["track_id"]
        for seg in entry["segments"]:
            seg_idx = seg.get("segment")
            key = (audio_file, seg_idx)
            per_segment[key].append((track_id, seg))

    results = {}  # nested results

    for (audio_file, seg_idx), track_list in per_segment.items():

        # sort tracks so ordering is consistent
        track_list.sort(key=lambda x: x[0])  # sort by track id
        track_ids = [tid for tid, _ in track_list]
        num_tracks = len(track_list)
        if num_tracks == 0:
            continue

        # collect cosine vectors, check length consistency
        cos_vectors = []
        start_ms = None
        for track_id, seg in track_list:
            per_window = seg.get("per_window", [])
            cos_vec = np.array([w["cosine"] for w in per_window], dtype=np.float32)
            cos_vectors.append(cos_vec)
            if start_ms is None:
                start_ms = seg.get("start_ms", 0)
                end_ms = seg.get("end_ms", None)

        lengths = [len(v) for v in cos_vectors]
        if len(set(lengths)) != 1:
            # mismatch: skip or pad/truncate — here we skip this segment
            print(f"WARNING: window length mismatch for {audio_file} segment {seg_idx}: {lengths}")
            continue
        T = lengths[0]

        # stack into [T, N_tracks]
        stack = np.stack(cos_vectors, axis=1)  # shape [T, N]
        logits = torch.from_numpy(stack)  # per-window logits

        with torch.no_grad():
            per_window_probs = stable_softmax(logits, temp=temperature).cpu().numpy()  # [T, N]

        # smoothing
        smooth_uniform = uniform_filter1d(per_window_probs, size=smooth_size, axis=0)
        smooth_ema = ema_smoothing(per_window_probs, alpha=ema_alpha)

        # Build time axis in seconds
        xs = (np.arange(T) * frame_step_ms + (start_ms or 0)) / 1000.0

        # Store results
        results.setdefault(audio_file, {})[seg_idx] = {
            "track_ids": track_ids,
            "per_window_probs": per_window_probs,          # raw softmax per window
            "smoothed_uniform": smooth_uniform,
            "smoothed_ema": smooth_ema,
            "time_s": xs,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "num_windows": T,
        }

        with open(pkl_output_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"✓ Saved probabilities to: {pkl_output_path}")

    return results
