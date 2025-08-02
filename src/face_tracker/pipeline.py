import os
import subprocess
import pickle
from shutil import rmtree
from .detect import inference_video, scene_detect
from .track import track_shot
from .crop import crop_video


def run_face_tracking(videofile_path, avi_dir, frames_dir, tmp_dir, work_dir, crop_dir, crop_scale, min_track,
                      facedet_scale, num_failed_det, min_face_size, frame_rate):
    """
    Runs full face tracking pipeline: video conversion, face detection, scene detection,
    tracking, and face crop extraction.

    :param videofile_path: str
        Path to the input video file.
    :param avi_dir: str
        Directory to store the standardized .avi version of the video.
    :param frames_dir: str
        Directory to store extracted video frames (.jpg).
    :param tmp_dir: str
        Temporary directory for intermediate files (will be deleted at end).
    :param work_dir: str
        Directory to store face and scene detection results (e.g. faces.pkl, scene.pkl).
    :param crop_dir: str
        Output directory for cropped face video tracks.
    :param crop_scale: float
        Padding scale for cropping around the face (e.g. 0.4 for 40% padding).
    :param min_track: int
        Minimum number of consecutive frames required to keep a track.
    :param facedet_scale: float
        Scale factor for the face detector input image.
    :param num_failed_det: int
        Maximum allowed skipped frames when linking detections in tracking.
    :param min_face_size: float
        Minimum average face size required to keep a track.
    :param frame_rate: float
        Frame rate of the video (used for output timing and crops).

    :return: None
        Results (cropped face tracks and metadata) are saved to disk as:
            - Cropped videos in `crop_dir`
            - Metadata pickle in `work_dir/tracks.pkl`
    """

    # Convert video to standardised avi and extract frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" %
               (videofile_path, os.path.join(avi_dir, 'video.avi')))
    subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" %
               (os.path.join(avi_dir, 'video.avi'), os.path.join(frames_dir, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)

    # face detection
    faces = inference_video(avi_dir, frames_dir, work_dir, facedet_scale)

    # scene detection
    scene = scene_detect(avi_dir, work_dir)

    # face tracking
    alltracks = []
    vidtracks = []

    for shot in scene:

      if shot[1].frame_num - shot[0].frame_num >= min_track:
        alltracks.extend(track_shot(min_track, num_failed_det, min_face_size, faces[shot[0].frame_num:shot[1].frame_num]))

    # face track crop
    for ii, track in enumerate(alltracks):
      vidtracks.append(crop_video(frames_dir, crop_scale, frame_rate, track,os.path.join(crop_dir, '%05d' % ii)))

    # save results
    savepath = os.path.join(work_dir, 'tracks.pkl')

    with open(savepath, 'wb') as fil:
      pickle.dump(vidtracks, fil)

    rmtree(tmp_dir)

