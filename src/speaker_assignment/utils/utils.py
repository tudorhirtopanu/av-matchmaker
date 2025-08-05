import pickle
import os
from collections import defaultdict


def load_pickle(path: str):
    """
    Loads a Python object from a pickle file.

    :param path: str
        Path to the pickle file.

    :return: object
        The deserialized Python object from the pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def group_segments_by_audio_and_track(flat):
    """
    Groups flat per-segment entries into a structured format by audio file and track ID.

    Each group will contain metadata and a list of segments with their respective details.

    :param flat: list
        A list of dictionaries, each representing a segment with keys such as:
            - 'audio_file': str
            - 'track_id': int
            - 'segment': int
            - 'start_ms': float
            - 'end_ms': float
            - 'num_windows': int
            - 'mean_cosine': float
            - 'per_window': list

    :return: list
        A list of grouped dictionaries, each with:
            - 'audio_file': str
            - 'track_id': int
            - 'segments': list of dicts
            - 'total_num_windows': int
            - 'overall_mean_cosine': float
    """
    grouped = defaultdict(lambda: {"audio_file": None, "track_id": None, "segments": []})

    for entry in flat:
        audio_file = entry.get("audio_file")
        track_id = entry.get("track_id")
        if audio_file is None or track_id is None:
            continue  # skip malformed

        key = (audio_file, track_id)
        container = grouped[key]
        container["audio_file"] = audio_file
        container["track_id"] = track_id

        segment_dict = {
            "segment": entry.get("segment"),
            "start_ms": entry.get("start_ms"),
            "end_ms": entry.get("end_ms"),
            "num_windows": entry.get("num_windows", 0),
            "mean_cosine": entry.get("mean_cosine", 0.0),
            "per_window": entry.get("per_window", []),
        }
        container["segments"].append(segment_dict)

    results = []
    for container in grouped.values():
        # Sort segments for deterministic order
        container["segments"].sort(key=lambda s: (s.get("segment", 0), s.get("start_ms", 0)))

        # Compute aggregated metadata
        total_windows = sum(s.get("num_windows", 0) for s in container["segments"])
        if total_windows > 0:
            weighted_sum = sum(
                s.get("mean_cosine", 0.0) * s.get("num_windows", 0)
                for s in container["segments"]
            )
            overall_mean = weighted_sum / total_windows
        else:
            overall_mean = 0.0

        grouped_entry = {
            "audio_file": container["audio_file"],
            "track_id": container["track_id"],
            "segments": container["segments"],
            "total_num_windows": total_windows,
            "overall_mean_cosine": overall_mean,
        }
        results.append(grouped_entry)

    return results


def load_and_group_av_cosine(input_pkl, output_pkl):
    """
    Loads a pickle file containing AV cosine similarity data and groups segments
    by audio file and track ID, computing summary metadata.

    If the input data is already grouped, it is returned as-is (with optional recomputation).
    The grouped data can optionally be saved to another pickle file.

    :param input_pkl: str
        Path to the input pickle file. Should contain either flat or grouped data.

    :param output_pkl: str
        Path to save the grouped output pickle file. If None, data is not saved.

    :return: list
        A list of grouped entries, each with:
            - 'audio_file': str
            - 'track_id': int
            - 'segments': list of dicts
            - 'total_num_windows': int
            - 'overall_mean_cosine': float
    """
    if not os.path.isfile(input_pkl):
        raise FileNotFoundError(f"Input pickle does not exist: {input_pkl}")

    data = load_pickle(input_pkl)

    # Detect already grouped format: list of dicts with 'segments'
    already_grouped = (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], dict)
        and "segments" in data[0]
        and "audio_file" in data[0]
        and "track_id" in data[0]
    )

    if already_grouped:
        grouped = data
        # Optionally, could recompute metadata here if needed
    elif isinstance(data, (list, tuple)):
        grouped = group_segments_by_audio_and_track(list(data))
    else:
        raise ValueError("Unexpected format in input pickle; expected flat list or grouped list.")

    if output_pkl:
        tmp_path = output_pkl + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(grouped, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, output_pkl)

    return grouped
