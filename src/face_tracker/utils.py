
def bb_intersection_over_union(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    :param boxA: list or tuple of float
        Bounding box in the format [x1, y1, x2, y2].
    :param boxB: list or tuple of float
        Bounding box in the format [x1, y1, x2, y2].

    :return: float
        IoU value between the two boxes (range: 0.0 to 1.0).
        Returns 0.0 if there is no overlap or if area is zero.
    """

    # Coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    # Compute area of each box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    iou = interArea / denom
    return float(iou)
