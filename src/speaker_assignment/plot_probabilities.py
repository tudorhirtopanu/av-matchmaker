import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pprint


def plot_probs(pkl_input_path, output_dir):
    """
    Plots per-speaker probability curves over time for each audio file segment.

    Reads smoothed speaker probabilities from a pickle file (e.g., output of `compute_probability`)
    and creates one PNG plot per audio file showing speaker likelihoods over time.

    :param pkl_input_path: str
        Path to the input pickle file containing computed speaker probabilities.
        Expected format is a nested dictionary:
            { audio_file: { segment_idx: { 'time_s': np.ndarray, 'smoothed_ema': np.ndarray, ... } } }

    :param output_dir: str
        Directory to save the output plot images as PNG files.
        One file will be saved per audio file (e.g., `filename_probs.png`).

    :return: None
        The function writes plots to disk and prints progress messages.
    """

    with open(pkl_input_path, 'rb') as f:
        data = pickle.load(f)

    pp = pprint.PrettyPrinter(indent=2, width=100)

    for audio_file, segments in data.items():
        print(f"\n=== Audio File: {audio_file} ===")
        track_ids = None
        all_probs = []
        all_times = []

        for seg_idx in sorted(segments.keys()):
            seg = segments[seg_idx]

            print(f"  Segment {seg_idx}:")
            print(f"    Track IDs: {seg['track_ids']}")
            print(f"    Start ms: {seg['start_ms']}")
            print(f"    End ms: {seg.get('end_ms')}")
            print(f"    Num windows: {seg['num_windows']}")

            if track_ids is None:
                track_ids = seg['track_ids']

            start_s = seg["start_ms"] / 1000.0
            end_s = seg.get("end_ms", None)
            time_s = seg["time_s"]
            probs = seg["smoothed_ema"]

            # Mask to only include times within start-end of segment
            mask = (time_s >= start_s) & (time_s <= end_s)
            seg_probs = probs[mask]
            seg_times = time_s[mask]

            all_probs.append(seg_probs)
            all_times.append(seg_times)

            # Append NaNs to break the plot between segments
            all_probs.append(np.full((1, seg_probs.shape[1]), np.nan))
            all_times.append(np.full((1,), np.nan))

        # Concatenate segments for plotting
        probs = np.concatenate(all_probs, axis=0)
        times = np.concatenate(all_times, axis=0)

        # Plot
        plt.figure(figsize=(10, 4))
        for i, tid in enumerate(track_ids):
            plt.plot(times, probs[:, i], label=f"Track {tid}")

        plt.xlabel("Time (s)")
        plt.ylabel("P(Speaking)")
        plt.title(f"Speaker Probabilities: {audio_file}")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()

        # Save plot
        fname = os.path.splitext(os.path.basename(audio_file))[0]
        save_path = os.path.join(output_dir, f"{fname}_probs.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"  ✓ Saved plot to: {save_path}")

