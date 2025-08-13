import os
import pickle
import cv2
from collections import defaultdict


def _draw_label(img, lines, x, y, font_scale=0.8, thickness=2, bg_color=(0, 0, 0), txt_color=(255, 255, 255)):
    """
    Draw a filled multi-line text label with bottom-left corner at (x, y).

    :param img: ndarray
        BGR image to draw on (modified in-place).
    :param lines: List[str]
        Text lines to render (top to bottom).
    :param x: int
        X position of label (pixels).
    :param y: int
        Y position of label (pixels).
    :param font_scale: float
        OpenCV font scale.
    :param thickness: int
        Text stroke thickness.
    :param bg_color: Tuple[int, int, int]
        Label background in BGR.
    :param txt_color: Tuple[int, int, int]
        Text color in BGR.

    :return: None
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]

    text_w = max(w for (w, _), _ in sizes)
    text_h = sum(h for (_, h), _ in sizes) + sum(b for _, b in sizes)

    pad = 4
    x2, y2 = x + text_w + 2 * pad, y + text_h + 2 * pad
    cv2.rectangle(img, (x, y), (x2, y2), bg_color, cv2.FILLED)

    y_cursor = y + pad + sizes[0][0][1]
    for t, ((_, h), b) in zip(lines, sizes):
        cv2.putText(img, t, (x + pad, y_cursor), font, font_scale, txt_color, thickness, cv2.LINE_AA)
        y_cursor += h + b


def _palette_color(track_id):
    """
    Get a deterministic BGR color for a track id.

    :param track_id: int
        Track identifier.

    :return: Tuple[int, int, int]
        Color in BGR.
    """
    palette = [
        (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
        (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 158, 158),
        (106, 27, 154), (0, 121, 107), (255, 87, 34), (3, 169, 244)
    ]
    return palette[track_id % len(palette)]


def _load_tracks_as_frame_map(tracks_pkl_path):
    """
    Load tracks.pkl and index boxes by frame.

    :param tracks_pkl_path: str
        Path to tracks.pkl.

    :return: Dict[int, List[Tuple[int, int, int, int, int]]]
        Map: frame_idx -> list of (x1, y1, x2, y2, track_id) with int pixel coords.
    """
    with open(tracks_pkl_path, "rb") as f:
        tracks = pickle.load(f)

    frame_to_items = defaultdict(list)
    for tid, t in enumerate(tracks):
        frames = t["track"]["frame"]
        bboxes = t["track"]["bbox"]
        # frames/bboxes may be numpy arrays
        for fi, box in zip(list(frames), list(bboxes)):
            x1, y1, x2, y2 = map(int, map(round, box))
            frame_to_items[int(fi)].append((x1, y1, x2, y2, tid))
    return frame_to_items


def _load_inv_assignment_map(assignments_path, mode):
    """
    Load assignments.pkl and build a map from track_id -> (audio_file, probability)
    using only the winning assignments from the given mode.

    :param assignments_path: str
        Path to assignments.pkl.
    :param mode: str
        Which assignment variant to use: 'raw', 'uniform', or 'ema'.
    :return: dict[int, (str, float)]
        Mapping: track_id -> (winning audio filename, probability score)
    """
    if not assignments_path:
        return {}
    with open(assignments_path, "rb") as f:
        A = pickle.load(f)

    if "assignment" not in A or mode not in A["assignment"]:
        raise ValueError(f"Mode '{mode}' not found in assignments.")

    audio_to_track = A["assignment"][mode]  # WAV -> tid
    inv_map = {}
    probs = A[mode]  # WAV -> {tid: prob}

    for wav, tid in audio_to_track.items():
        p = probs[wav].get(tid, probs[wav].get(str(tid), 0.0))
        inv_map[int(tid)] = (wav, float(p))

    return inv_map


def annotate_video(video_path, tracks_path, output_video_path, assignments_path=None, mode='raw', fixed_box_color=None,
                   font_scale=0.8, thickness=2):
    """
    Draw track boxes and audio labels onto a video and save it.

    :param video_path: str
        Input video (use the same one used for tracking).
    :param tracks_path: str
        Path to tracks.pkl.
    :param output_video_path: str
        Output annotated video path (.mp4/.avi).
    :param assignments_path: Optional[str]
        Path to assignments.pkl (optional).
    :param mode: Optional[str]
        Assignment variant: 'raw'|'uniform'|'ema'.
    :param fixed_box_color: Optional[Tuple[int, int, int]]
        Single BGR color; if None, use per-track palette.
    :param font_scale: float
        Label font scale.
    :param thickness: int
        Box/text thickness.

    :return: None
    """
    frame_to_items = _load_tracks_as_frame_map(tracks_path)
    inv_assign = _load_inv_assignment_map(assignments_path, mode)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output_video_path.lower().endswith(".mp4") else "XVID"))
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for x1, y1, x2, y2, tid in frame_to_items.get(frame_idx, []):

            # clamp
            x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
            color = fixed_box_color if fixed_box_color is not None else _palette_color(tid)

            # rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # label lines
            audio_file, conf = inv_assign.get(tid, (None, None))
            if audio_file is None:
                lines = [f"track {tid}"]
            else:
                lines = [f"track {tid}", f"{os.path.basename(audio_file)} ({conf:.2f})"]

            # position label above box if possible, else below
            label_x = x1
            label_y = y1 - 6 - int(18 * font_scale)
            if label_y < 0:
                label_y = y2 + 6
            _draw_label(frame, lines, label_x, label_y, font_scale, thickness, bg_color=color, txt_color=(0,0,0))

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {output_video_path}")
