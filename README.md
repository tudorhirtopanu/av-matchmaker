# AV Matchmaker

AV Matchmaker labels separated speaker audio with the identities of detected speakers in a video using SyncNet's cross-modal embeddings. It leverages the fact that audio activity and mouth movement are temporally correlated.

This project was built during the **Google DeepMind Research Ready Programme** (hosted at **Queen Mary University of London**) to address a limitation in the paper [Visual-based Spatial Audio Generation System for Multi-Speaker Environments](https://arxiv.org/abs/2502.07538) by Liu et al. (2025). In their pipeline for generating spatialized audio from mono recordings, each separated audio track had to be manually labeled with the corresponding face in the video.

This repository automates that manual labeling step by matching face tracks to their corresponding audio tracks, using a pretrained SyncNet model for visual and audio embeddings.

*(Independent research code)*

<img width="500" alt="Screenshot 2025-08-12 at 19 05 59" src="https://github.com/user-attachments/assets/0d59fa54-b173-413b-a2f1-07ccf77557f3" />

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Acknowledgements](#acknowledgements)

## Features 
- Matches multiple face tracks to separated audio tracks without manual labeling.
- Uses SyncNet’s visual–audio embedding space for speaker identification.
- Handles overlapping speakers and multiple faces in real-world videos.

## Installation

1. Clone this repository
```
git clone https://github.com/tudorhirtopanu/av-matchmaker.git
cd av-matchmaker
```

2. Create a virtual environment (recommended) and activate it
```
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Set up necessary directories and download weights
```
python3 setup.py
```

## Usage

The easiest way to run AV Matchmaker is via the `run_pipeline` script.  
This script handles the full process: face detection, tracking, audio processing, embedding extraction, matching, and visualization.  

```bash
./run_pipeline \
  --video_file /path/to/video.mp4 \
  --reference sessionName \
  --audio_dir /path/to/separated_audio
```

**Arguments (required):**
- `--video_file` — Path to the video file.
- `--reference` — Session identifier (used to name output subfolders).
- `--audio_dir` — Path to the folder containing separated `.wav` audio tracks (one per speaker).

**Optional arguments (defaults shown):**
- `--output_dir` — Root output directory (default: `<repo_root>/output`).
- `--facedet_scale` — Scale factor for face detection (default: `0.25`).
- `--crop_scale` — Padding scale around detected faces (default: `0.40`).
- `--min_track` — Minimum face track length in frames (default: `100`).
- `--num_failed_det` — Allowed missed detections in tracking (default: `25`).
- `--min_face_size` — Minimum average face size in pixels (default: `100`).
- `--frame_rate` — Frame rate of the video (default: `25.0`).
- `--weights_path` — Path to SyncNet model weights (default: provided model in repo).

### Expected outputs

After running the pipeline, the main outputs are:

- **Annotated video** (`/graphs/<reference>/annotated.mp4`)  
  Shows bounding boxes around detected faces, labeled with the face track ID and the matched audio file.

- **Probability matrix**  
  A matrix where **rows** = audio tracks and **columns** = face tracks, with each cell showing the match probability between that audio and face.

- **Assignments file** (`/pywork/<reference>/assignments.pkl`)  
  A pickle file containing detailed assignment scores for each audio–face pair under different weighting schemes (`raw`, `uniform`, `ema`).  
  Example:
  ```json
  {
      "raw": {
          "floyd.wav": {"0": 0.8368, "1": 0.1631},
          "host.wav": {"0": 0.2022, "1": 0.7977}
      },
      "uniform": {
          "floyd.wav": {"0": 0.7875, "1": 0.2124},
          "host.wav": {"0": 0.2282, "1": 0.7717}
      },
      "ema": {
          "floyd.wav": {"0": 0.8154, "1": 0.1846},
          "host.wav": {"0": 0.2840, "1": 0.7159}
      },
      "all_tids": [0, 1],
      "assignment": {
          "raw": {"floyd.wav": 0, "host.wav": 1},
          "uniform": {"floyd.wav": 0, "host.wav": 1},
          "ema": {"floyd.wav": 0, "host.wav": 1}
      }
  }
  ```

## Pipeline
<img width="633" height="463" alt="image" src="https://github.com/user-attachments/assets/eaf66d04-8398-4625-81c4-68c02127d459" />

### How it Works (Overview)

1. **Inputs:** Video (25 fps) and separated audio files for each speaker.  
2. **Face Tracking:** Detect faces (S3FD), track across frames (IoU), smooth positions, crop to 224×224 clips.  
3. **Audio Processing:** Apply VAD, segment speech into time windows aligned with video frames.  
4. **Embeddings:** Extract visual (face) and audio (speech) embeddings using SyncNet.  
5. **Matching:** Compare embeddings via cosine similarity, aggregate over time using weighted log probabilities.  
6. **Assignment:** Assign each audio track to the most likely face track based on aggregated probabilities.  
7. **Visualization:** Annotate video with face boxes and audio labels, and save a table of matches with confidence scores.

## Acknowledgements
This project uses the [PyTorch implementation of SyncNet](https://github.com/joonson/syncnet_python) for extracting cross-modal audio–visual embeddings. 

We also acknowledge the original SyncNet paper:

Chung, J.S. and Zisserman, A. (2016).  
*Out of time: automated lip sync in the wild*. Workshop on Multi-view Lip-reading, ACCV.  
[Paper link](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/)

