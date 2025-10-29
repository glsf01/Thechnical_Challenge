import os


def get_latest_model(detections_path: str) -> str:
    """
    Returns the latest model from a folder
    :param detections_path: Path to folder containing models
    :return: Name of the latest model
    """
    return os.listdir(detections_path)[-1]


def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes ([x1, y1, x2, y2])
    :param boxA: Bounding box [x1, y1, x2, y2]
    :param boxB: Bounding box [x1, y1, x2, y2]
    :return: Intersection over Union (IoU) between boxA and boxB
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0


if __name__ == '__main__':
    path = "../../runs/detect"
    get_latest_model(path)
