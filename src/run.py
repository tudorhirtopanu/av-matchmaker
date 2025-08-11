#!/usr/bin/env python3

import os
import argparse
from shutil import rmtree
from face_tracker import run_face_tracking
from audio_processing import run_audio_processing
from speaker_assignment import run_speaker_assignment

# default output directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_weights = os.path.join(models_dir, 'syncnet_v2.model')


def parse_args():
    # ===============================
    # ----- Parse the arguments -----
    # ===============================
    parser = argparse.ArgumentParser()

    # ---------------------------
    # Args that must be specified
    # ---------------------------
    parser.add_argument('--video_file', type=str, required=True, help='Path to the video file')
    parser.add_argument('--reference', type=str, required=True, help='Session Identifier (subfolder name)')

    # ------------------
    # Pre-specified args
    # ------------------
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='Root output directory containing pyavi, pytmp, pywork, pycrop, pyframes')

    # -----------------------------
    # Face detection/tracking args
    # -----------------------------
    parser.add_argument('--facedet_scale', type=float, default=0.25, help='Scale factor for face detection')
    parser.add_argument('--crop_scale', type=float, default=0.40, help='Padding scale around detected face')
    parser.add_argument('--min_track', type=int, default=100, help='Minimum facetrack length in frames')
    parser.add_argument('--num_failed_det', type=int, default=25, help='Allowed missed detections in tracking')
    parser.add_argument('--min_face_size', type=int, default=100, help='Minimum average face size in pixels')
    parser.add_argument('--frame_rate', type=float, default=25.0, help='Frame rate of the video')

    # ---------
    # VAD args
    # ---------
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to the .wav files')

    # ------------------------
    # Speaker Assignment args
    # ------------------------
    parser.add_argument('--weights_path', type=str, default=model_weights,
                        help='Path to the SyncNet model weights')

    return parser.parse_args()

def main():
    opt = parse_args()

    # Derived directories
    avi_dir = os.path.join(opt.output_dir, 'pyavi', opt.reference)
    frames_dir = os.path.join(opt.output_dir, 'pyframes', opt.reference)
    tmp_dir = os.path.join(opt.output_dir, 'pytmp', opt.reference)
    work_dir = os.path.join(opt.output_dir, 'pywork', opt.reference)
    crop_dir = os.path.join(opt.output_dir, 'pycrop', opt.reference)
    audio_dir = os.path.join(opt.output_dir, 'pyaudio', opt.reference)
    graphs_dir = os.path.join(opt.output_dir, 'graphs', opt.reference)

    # =============================
    # ----- Run the Pipeline -----
    # =============================

    # Clean and recreate directories (session specific)
    for d in [avi_dir, frames_dir, tmp_dir, crop_dir, work_dir, audio_dir]:
        if os.path.exists(d):
            rmtree(d)
        os.makedirs(d, exist_ok=True)

    # ----------------------------
    # Face Detection and Tracking
    # ----------------------------
    run_face_tracking(opt.video_file, avi_dir, frames_dir, tmp_dir, work_dir, crop_dir, opt.crop_scale, opt.min_track,
                      opt.facedet_scale, opt.num_failed_det, opt.min_face_size, opt.frame_rate)

    # ----------------------------
    # Audio Processing
    # ----------------------------
    run_audio_processing(work_dir, opt.audio_dir)

    # -------------------
    # Speaker Assignment
    # -------------------
    run_speaker_assignment(work_dir, opt.audio_dir, tmp_dir, crop_dir, graphs_dir, opt.weights_path, avi_dir)


if __name__ == "__main__":
    main()
