import numpy as np
from scipy.interpolate import interp1d

from .utils import bb_intersection_over_union


def track_shot(min_track, num_failed_det, min_face_size, scenefaces):
    """
    Tracks faces across frames within a scene by linking detections based on IOU and frame continuity.

    :param min_track: int
        Minimum number of consecutive frames required to keep a track.
    :param num_failed_det: int
        Maximum allowed number of skipped frames between consecutive detections.
    :param min_face_size: float
        Minimum average face size (in pixels) required to keep a track.
    :param scenefaces: list
        A list of per-frame face detections for a scene. Each element is a list of detections for one frame,
        where each detection is a dict with:
            - 'frame': int, frame index
            - 'bbox' : list of [x1, y1, x2, y2] bounding box coordinates

    :return: list
        A list of valid face tracks. Each track is a dict with:
            - 'frame': ndarray of frame indices
            - 'bbox' : ndarray of interpolated bounding boxes for each frame [N x 4]
    """

    iou_thresh = 0.5  # Minimum IOU between consecutive face detections
    tracks = []

    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iou_thresh:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break

        if track == []:
            break
        elif len(track) > min_track:

            framenum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])

            frame_i = np.arange(framenum[0], framenum[-1] + 1)

            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)

            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})

    return tracks
