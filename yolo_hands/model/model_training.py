"""
train_hands_yolo.py

Train a YOLO model (Ultralytics) on the provided YOLO-format dataset.
Configure parameters in the CONFIG dict below and run the script.

Example:
    # edit CONFIG paths/params, then
    python src/train_hands_yolo.py
"""

from ultralytics import YOLO
import os

# ---------- CONFIG ----------
CONFIG = {
    "data_yaml": os.path.abspath("../custom_hands_dataset/data.yaml"),
    "pretrained_model": "yolo11n.pt",
    "epochs": 20,
    "imgsz": 640,
    "batch": 16,
    "experiment_name": "hands_exp"
}


# ----------------------------

def main():
    model = YOLO(CONFIG["pretrained_model"])
    print("Starting YOLO training with config:", CONFIG)
    model.train(
        data=CONFIG["data_yaml"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        name=CONFIG["experiment_name"],
        device=0
    )
    print("Training finished. Check runs/detect/ for outputs.")


if __name__ == "__main__":
    main()
