import os
import pickle
import numpy as np


def load_probabilities(path):
    """
    Load a pickled data structure from disk.

    :param path: str
        Filesystem path to the pickle file.
    :return:
        The Python object stored in the pickle.
    :raises FileNotFoundError:
        If `path` does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def aggregate_multi_speaker_probs(per_window_probs, gap_exponent, sigmoid_temp, eps):
    """
    Aggregate per-window speaker probabilities into a single overall probability vector.

    At each time window t you have an N-dimensional probability vector p_t. This routine:
      1. Clips probabilities to avoid log(0).
      2. Computes a confidence weight per window based on how far the top speaker's probability
         is above the uniform baseline.
      3. Forms a weighted average of the log-probabilities across all windows.
      4. Applies a temperature scaling and final softmax to yield an N-vector of speaker likelihoods.

    :param per_window_probs: np.ndarray, shape (T, N)
        Raw per-window probability estimates for N speakers over T time windows.
    :param gap_exponent: float
        Exponent applied to the “gap” from uniform confidence to sharpen or flatten the weights.
    :param sigmoid_temp: float
        Temperature divisor for the final softmax. Higher values produce a flatter distribution.
    :param eps: float,
        Small constant to clip probabilities away from 0/1 for numerical stability.

    :return: np.ndarray, shape (N,)
        Final aggregated speaker probability vector summing to 1.
    """

    # Clip probabilities for numerical safety
    p = np.clip(per_window_probs, eps, 1 - eps)  # shape [T, N]
    N = p.shape[1]

    # Compute confidence weight per window:
    # how much the highest-prob speaker exceeds the uniform probability 1/N
    uniform = 1.0 / N
    top_minus_uniform = np.max(p, axis=1) - uniform  # [T,]
    w = top_minus_uniform ** gap_exponent # apply exponent to sharpen/flatten
    if w.sum() == 0:
        # if every window is at or below uniform, fall back to a uniform assignment
        return np.ones(N) / N

    # Compute weighted average of per-window log-probabilities
    log_p = np.log(p)  # natural log, shape [T, N]

    # multiply each row by its weight, sum, then normalize by total weight
    avg_logp = (w[:, None] * log_p).sum(axis=0) / w.sum()  # shape [N,]

    # Temperature scaling + softmax
    scaled = avg_logp / sigmoid_temp

    # subtract max for numeric stability
    ex = np.exp(scaled - np.max(scaled))
    return ex / ex.sum()

def compute_face_audio_matrix(data, gap_exponent, sigmoid_temp, eps):
    """
    Build face×audio probability matrices from per-window, multi-speaker probabilities.

    Iterates over each audio file in `data`, aggregates raw, uniform-smoothed, and EMA-smoothed
    per-window probabilities into a single probability per face-track, and returns three matrices.

    :param data: dict
        Nested structure:
            {
                audio_file_path: {
                    segment_idx: {
                        'track_ids': List[int],           # face-track identifiers in this segment
                        'per_window_probs': np.ndarray,   # shape [T_s, N] raw probs per window
                        'smoothed_uniform': np.ndarray,   # shape [T_s, N] uniform-smoothed
                        'smoothed_ema': np.ndarray        # shape [T_s, N] EMA-smoothed
                    },
                    ...
                },
                ...
            }

    :return:
        raw_matrix: dict[audio_file_path → dict[track_id → float]]
            Aggregated raw probabilities per track.
        uniform_matrix: dict[audio_file_path → dict[track_id → float]]
            Aggregated uniform-smoothed probabilities per track.
        ema_matrix: dict[audio_file_path → dict[track_id → float]]
            Aggregated EMA-smoothed probabilities per track.
        all_tids: List[int]
            Sorted list of all track IDs seen across all audio files.
    """

    # Sort audio file keys and gather all unique track IDs across the dataset
    audio_files = sorted(data.keys())
    all_tids = sorted({tid
                       for segments in data.values()
                       for seg in segments.values()
                       for tid in seg["track_ids"]})

    # Initialize output matrices (per-audio-file dictionaries)
    raw_matrix = {a: {} for a in audio_files}
    uniform_matrix = {a: {} for a in audio_files}
    ema_matrix = {a: {} for a in audio_files}

    # Process each audio file independently
    for audio_file in audio_files:
        segments = data[audio_file]
        # assume same ordering of track_ids in each segment
        track_ids = next(iter(segments.values()))["track_ids"]

        # Stack per-window probability arrays across all segments
        raw_list = [segments[s]["per_window_probs"] for s in sorted(segments)]
        uni_list = [segments[s]["smoothed_uniform"] for s in sorted(segments)]
        ema_list = [segments[s]["smoothed_ema"] for s in sorted(segments)]

        raw_mat = np.vstack(raw_list)  # shape [T_total, N]
        uni_mat = np.vstack(uni_list)
        ema_mat = np.vstack(ema_list)

        # Aggregate each matrix into a single N-vector of speaker probs
        p_raw = aggregate_multi_speaker_probs(raw_mat, gap_exponent, sigmoid_temp, eps)
        p_uniform = aggregate_multi_speaker_probs(uni_mat, gap_exponent, sigmoid_temp, eps)
        p_ema = aggregate_multi_speaker_probs(ema_mat, gap_exponent, sigmoid_temp, eps)

        # Assign aggregated probabilities to each track ID present
        for idx, tid in enumerate(track_ids):
            raw_matrix[audio_file][tid] = float(p_raw[idx])
            uniform_matrix[audio_file][tid] = float(p_uniform[idx])
            ema_matrix[audio_file][tid] = float(p_ema[idx])

        # For any track not in this file, explicitly set its probability to zero
        for other in all_tids:
            if other not in track_ids:
                raw_matrix[audio_file].setdefault(other, 0.0)
                uniform_matrix[audio_file].setdefault(other, 0.0)
                ema_matrix[audio_file].setdefault(other, 0.0)

    # Return the three probability matrices and the list of all track IDs
    return raw_matrix, uniform_matrix, ema_matrix, all_tids


def print_matrix(title, matrix, all_tids):
    """
    Print a face×audio probability matrix to the console.

    :param title: str
        Heading to display above the table.
    :param matrix: dict[str, dict[int, float]]
        Nested dict mapping audio file paths to dicts of track_id→probability.
    :param all_tids: list[int]
        Sorted list of all track IDs to show as columns.
    :return: None
        Prints formatted table directly to stdout.
    """
    print(f"\n=== {title} ===")
    header = "Audio File".ljust(20) + "".join(f"{tid:^10}" for tid in all_tids)
    print(header)
    print("-" * len(header))
    for audio_file, row in matrix.items():
        name = os.path.basename(audio_file)
        line = name.ljust(20) + "".join(f"{row.get(tid,0):10.3f}" for tid in all_tids)
        print(line)
    print()


def assign(probabilities_path, output_path, gap_exponent=45, sigmoid_temp=2.0, eps = 1e-8):
    """
    Load per-window speaker probabilities, aggregate them into matrices,
    display those matrices, compute a per-audio-file speaker assignment,
    and save all results to a pickle file.

    :param probabilities_path: str
        Path to the input pickle file containing per-window speaker probabilities.
    :param output_path: str
        Path where the output pickle (with matrices and assignments) will be saved.
    :param gap_exponent: float, default=45
        Exponent applied to the confidence gap when aggregating per-window log-probs.
    :param sigmoid_temp: float, default=2.0
        Temperature divisor for the final softmax step.
    :param eps: float, default=1e-8
        Small constant for clipping probabilities away from 0 and 1.

    :return: None
        Outputs are printed to stdout and written to disk; nothing is returned.
    """

    # Load and compute
    data = load_probabilities(probabilities_path)
    raw_mat, uni_mat, ema_mat, all_tids = compute_face_audio_matrix(data, gap_exponent, sigmoid_temp, eps)

    # print for the user
    print_matrix("Raw Aggregated Probabilities", raw_mat, all_tids)
    print_matrix("Uniform-Weighted Probabilities", uni_mat, all_tids)
    print_matrix("EMA-Weighted Probabilities", ema_mat, all_tids)

    # build assignments
    assignment = {
        "raw":     {audio: max(row, key=row.get) for audio, row in raw_mat.items()},
        "uniform": {audio: max(row, key=row.get) for audio, row in uni_mat.items()},
        "ema":     {audio: max(row, key=row.get) for audio, row in ema_mat.items()},
    }

    # package everything into one dict
    out = {
        "raw":        raw_mat,
        "uniform":    uni_mat,
        "ema":        ema_mat,
        "all_tids":   all_tids,
        "assignment": assignment,
    }

    # write to pickle
    with open(output_path, "wb") as f:
        pickle.dump(out, f)

