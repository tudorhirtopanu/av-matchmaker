import os
import glob
import cv2
import numpy as np
from scipy import signal


def crop_video(frames_dir, crop_scale, frame_rate, track, cropfile):
    """
    Crops and smooths face regions from video frames based on a face track, and saves the result as a video.

    :param frames_dir: str
                Directory containing extracted video frames (.jpg).

    :param crop_scale: float
                Padding around the face bounding box (e.g. 0.4 adds 40%).

    :param frame_rate: int
                Frame rate of the video (default 25fps)

    :param track: dict
                Face tracking data with:
                    - 'frame': list of frame indices
                    - 'bbox' : list of [x1, y1, x2, y2] boxes per frame.

    :param cropfile: str
                Output filename prefix for the cropped face video.

    :return:
                dict
                    {
                        'track': track,               # Original track data
                        'proc_track': dict,          # Smoothed face centers/sizes
                        'video_path': str,           # Path to output .avi video
                        'start_sec': float,          # Start time in seconds
                        'end_sec': float             # End time in seconds
                    }
    """

    # Gather and sort the frame image list
    flist = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    face_vid = f"{cropfile}.avi"
    vOut = cv2.VideoWriter(face_vid, fourcc, frame_rate, (224, 224))

    # Compute centers and sizes, then smooth
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)  # crop center x
        dets['x'].append((det[0] + det[2]) / 2)  # crop center y

    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    # write each frame's crop
    for fidx, frame in enumerate(track['frame']):
        cs = crop_scale
        bs = dets['s'][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        image = cv2.imread(flist[frame])

        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X

        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]

        vOut.write(cv2.resize(face, (224, 224)))
    vOut.release()

    # Compute time window (seconds)
    start_sec = track['frame'][0] / frame_rate
    end_sec = (track['frame'][-1] + 1) / frame_rate

    return {
        'track': track,
        'proc_track': dets,
        'video_path': face_vid,
        'start_sec': start_sec,
        'end_sec': end_sec
    }
