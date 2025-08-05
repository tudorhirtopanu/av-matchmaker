import os
import cv2
import numpy as np
import glob
import subprocess
import torch


def extract_video_tensor(face_clip, temp_dir):
    """
    Extracts frames from a video clip and converts them into a SyncNet-compatible input tensor.

    :param face_clip: str
        Path to the input video file (typically a cropped face video).

    :param temp_dir: str
        Temporary directory to store extracted video frames as JPEGs.

    :return: tuple
        A tuple (imtv, num_frames) where:
            - imtv (torch.Tensor): 5D tensor of shape [1, C, N, H, W] representing the video.
            - num_frames (int): Number of frames extracted from the video.

        Returns (None, 0) if no frames are found or extraction fails.
    """
    if os.path.exists(temp_dir):
        subprocess.call(f"rm -rf {temp_dir}", shell=True)
    os.makedirs(temp_dir, exist_ok=True)

    cmd = f"ffmpeg -y -i {face_clip} -threads 1 -f image2 {temp_dir}/%06d.jpg"
    subprocess.call(cmd, shell=True)

    flist = sorted(glob.glob(os.path.join(temp_dir, '*.jpg')))
    if not flist:
        return None, 0

    images = [cv2.imread(fn) for fn in flist]
    im_np = np.stack(images, axis=3)                # H×W×C×N
    im_np = np.expand_dims(im_np, axis=0)           # 1×H×W×C×N
    im_np = np.transpose(im_np, (0, 3, 4, 1, 2))    # 1×C×N×H×W
    imtv = torch.from_numpy(im_np.astype(float)).float()
    num_frames = len(images)

    return imtv, num_frames
