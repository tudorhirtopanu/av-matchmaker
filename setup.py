import os
import urllib.request

# Make directories
os.makedirs("models", exist_ok=True)
os.makedirs("detectors/s3fd/weights", exist_ok=True)

# Download SyncNet model
urllib.request.urlretrieve(
    "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
    "models/syncnet_v2.model"
)

# Download S3FD model
urllib.request.urlretrieve(
    "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth",
    "detectors/s3fd/weights/sfd_face.pth"
)
