import os
import pickle

import torch
import torch.nn.functional as F

from .utils import extract_audio_segments, extract_video_tensor
from src.syncnet import SyncNetInstance


def compute_cosine_similarity(work_dir, audio_dir, temp_dir, crop_dir, weights_path, speech_segments_pkl, face_tracks_pkl, batch_size):
    """
    Computes cosine similarity between audio segments and face video tracks using SyncNet.

    For each face track and each audio segment, this function:
      - Loads and preprocesses audio and video data.
      - Runs SyncNet to compute audio and visual embeddings.
      - Calculates per-window and mean cosine similarity between audio and visual features.

    Results are saved as a list of dictionaries to 'av_cosine.pkl' in the work directory.

    :param work_dir: str
        Directory where the output 'av_cosine.pkl' file will be saved.

    :param audio_dir: str
        Directory containing original WAV audio files.

    :param temp_dir: str
        Temporary directory used for intermediate audio and video frame extraction.

    :param crop_dir: str
        Directory containing cropped face track video files named as '{track_id:05d}.avi'.

    :param weights_path: str
        Path to the SyncNet model weights (.pth file).

    :param speech_segments_pkl: str
        Path to the pickle file containing speech segment metadata (segment map).

    :param face_tracks_pkl: str
        Path to the pickle file containing face tracking information.

    :param batch_size: int
        Batch size used when running inference over the SyncNet model.

    :return: None
        The function saves the output to disk and prints status logs.
    """

    # Load the instance and the weights
    s = SyncNetInstance()
    s.loadParameters(weights_path)
    s.eval()

    # Open speech windows pkl file
    with open(speech_segments_pkl, 'rb') as f:
        segment_map = pickle.load(f, encoding='latin1')

    audio_data = extract_audio_segments(segment_map, audio_dir, temp_dir)

    # Open face tracks pkl file
    with open(face_tracks_pkl, 'rb') as f:
        tracks = pickle.load(f, encoding='latin1')

    all_results = []

    for idx, track in enumerate(tracks):
        print(f"\n Processing face‐track {idx}")

        face_clip = os.path.join(crop_dir, f'{idx:05d}.avi')

        if not os.path.exists(face_clip):
            print(" Missing clip:", face_clip)
            continue

        # Extract video tensor
        temp_frame_dir = os.path.join(temp_dir, f'track_{idx:05d}')
        imtv, num_fr = extract_video_tensor(face_clip, temp_frame_dir)

        if imtv is None or num_fr == 0:
            print(f"No frames for track {idx}, skipping.")
            continue

        for a in audio_data:

            # align each precomputed segment
            num_samp = a['num_samp']
            min_len = min(num_fr, num_samp)
            last = min_len - 5

            # run SyncNet forward passes to get embeddings
            im_feats = []
            cc_feats = []

            for i in range(0, last, batch_size):
                end = min(last, i + batch_size)

                # video windows: each is 5 consecutive frames
                im_batch = [imtv[:, :, f: f + 5, :, :] for f in range(i, end)]
                im_in = torch.cat(im_batch, 0)
                with torch.no_grad():
                    im_out = s.S.forward_lip(im_in)  # [B,1024]
                im_feats.append(im_out.cpu())

                # audio windows: each is 20 MFCC frames (5×4)
                cc_batch = [a['cct'][:, :, :, f * 4: f * 4 + 20] for f in range(i, end)]
                cc_in = torch.cat(cc_batch, 0)
                with torch.no_grad():
                    cc_out = s.S.forward_aud(cc_in)  # [B,1024]
                cc_feats.append(cc_out.cpu())

            # concatenate all batches
            im_feat = torch.cat(im_feats, 0)  # [num_windows, 1024]
            cc_feat = torch.cat(cc_feats, 0)  # [num_windows, 1024]

            # per‑window cosine similarity & mean
            sims = F.cosine_similarity(cc_feat, im_feat, dim=1)  # [num_windows]
            frame_idxs = list(range(sims.shape[0]))

            per_window = [
                {'frame': frame_idxs[i], 'cosine': float(sims[i].item())}
                for i in range(len(frame_idxs))
            ]
            mean_cosine = float(sims.mean().item())

            print(f"  ✓ Track {idx}, {a['fname']}[seg#{a['seg_idx']}] → "
                  f"{len(per_window)} windows, mean cosine {mean_cosine:.3f}")

            all_results.append({
                'track_id': idx,
                'audio_file': a['fname'],
                'segment': a['seg_idx'],
                'start_ms': a['start_ms'],
                'end_ms': a['end_ms'],
                'num_windows': len(per_window),
                'mean_cosine': mean_cosine,
                'per_window': per_window
            })

    # Save results
    out_pkl = os.path.join(work_dir, 'av_cosine.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll done! Saved results to {out_pkl}")

