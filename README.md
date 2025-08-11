# AV Matchmaker

AV Matchmaker labels separated speaker audio with the identities of detected speakers in a video using SyncNet's cross-modal embeddings. It leverages the fact that audio activity and mouth movement are temporally correlated.

This project was built during the **Google DeepMind Research Ready Programme** (hosted at **Queen Mary University of London**) to address a limitation in the paper [Visual-based Spatial Audio Generation System for Multi-Speaker Environments](https://arxiv.org/abs/2502.07538) by Liu et al. (2025). In their pipeline for generating spatialized audio from mono recordings, each separated audio track had to be manually labeled with the corresponding face in the video.

This repository automates that manual labeling step by matching face tracks to their corresponding audio tracks, using a pretrained SyncNet model for visual and audio embeddings.

*(Independent research code)*

## Table of Contents
- [Features](#features)
- [How it Works (Overview)](#how-it-works-overview)

## Features 
- Matches multiple face tracks to separated audio tracks without manual labeling.
- Uses SyncNet’s visual–audio embedding space for speaker identification.
- Handles overlapping speakers and multiple faces in real-world videos.

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
