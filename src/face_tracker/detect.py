import os
import glob
import pickle
import time

import cv2
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from detectors import S3FD


def inference_video(avi_dir, frames_dir, work_dir, facedet_scale):
    """
    Runs face detection on all frames in a video directory using the S3FD detector.

    :param avi_dir: str
        Directory containing the source video (used for logging).
    :param frames_dir: str
        Directory containing extracted video frames as .jpg files.
    :param work_dir: str
        Directory to save the resulting detections (faces.pkl).
    :param facedet_scale: float
        Image scale factor for the face detector (affects speed vs. accuracy).

    :return: list
        A list of per-frame detections, where each element is a list of dicts with:
            - 'frame': frame index (int)
            - 'bbox' : bounding box [x1, y1, x2, y2] (list of float)
            - 'conf' : confidence score (float)
    """

    DET = S3FD(device='cpu')

    flist = glob.glob(os.path.join(frames_dir, '*.jpg'))
    flist.sort()

    dets = []

    for fidx, fname in enumerate(flist):

        start_time = time.time()

        image = cv2.imread(fname)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])

        dets.append([]);
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})

        elapsed_time = time.time() - start_time

        print('%s-%05d; %d dets; %.2f Hz' % (
        os.path.join(avi_dir, 'video.avi'), fidx, len(dets[-1]), (1 / elapsed_time)))

    savepath = os.path.join(work_dir, 'faces.pkl')

    with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)

    return dets


def scene_detect(avi_dir, work_dir):
    """
    Performs scene change detection on a video.

    :param avi_dir: str
        Directory containing the input video file ('video.avi').
    :param work_dir: str
        Directory to save the detected scenes (scene.pkl).

    :return: list
        A list of scene boundary tuples, each as (start_timecode, end_timecode).
    """

    video_manager = VideoManager([os.path.join(avi_dir, 'video.avi')])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()

    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode)

    savepath = os.path.join(work_dir, 'scene.pkl')

    if scene_list == []:
      scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

    with open(savepath, 'wb') as fil:
      pickle.dump(scene_list, fil)

    print('%s - scenes detected %d' % (os.path.join(avi_dir, 'video.avi'), len(scene_list)))

    return scene_list

