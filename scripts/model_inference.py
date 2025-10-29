import argparse

import cv2 as cv
from ultralytics import YOLO
from utils.utils_aux import get_latest_model
from scripts.finite_state_machine import OperationFSM


def run_yolo_inference(model_path: str, video_path: str, window_name: str = "YOLO Inference"):
    """
    Run YOLO inference on a video and display annotated frames.
    :param model_path: Path to the YOLO model weights (.pt file).
    :param video_path: Path to the input video file.
    :param window_name: Name of the OpenCV display window.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        """
        # Code segment to identify the contents of bounding boxes
        print(results[0].names)
        for box in results[0].boxes:
            # Bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            xywh = box.xywh[0].cpu().numpy()  # [x_center, y_center, w, h]

            # Confidence and class
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]

            print(f"{label}: conf={conf:.2f}, xyxy={xyxy.flatten().shape}, xywh={xywh}")
        print("----------------------------------- END OF FRAME -----------------------------------")
        return
        """
        # Display the annotated frame
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()


def run_yolo_fsm_inference(model_path: str, video_path: str, display_predictions: bool = False, window_name: str = "YOLO Inference"):
    """
    Run YOLO inference on a video, update FSM, and optionally display annotated frames.
    :param model_path: Path to the YOLO model weights (.pt file).
    :param video_path: Path to the input video file.
    :param display_predictions: If True, displays annotated frames in a window.
    :param window_name: Name of the OpenCV display window.
    """
    model = YOLO(model_path)
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fsm = OperationFSM(iou_thresh=0.1)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        results = model(
            frame,
            verbose=False,
        )

        hands, pieces, probes, markers = [], [], [], []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]

            if label == "hand":
                hands.append(xyxy)
            elif label == "piece":
                pieces.append(xyxy)
            elif label == "probe":
                probes.append(xyxy)
            elif label == "marker":
                markers.append(xyxy)

        event = fsm.update(hands, pieces, probes, markers, frame_idx, fps)

        if display_predictions:
            if event:
                print(f"Frame {frame_idx}: Event = {event}")
            annotated_frame = results[0].plot()
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.imshow(window_name, annotated_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()
    print("Final metrics:", fsm.summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="../runs/detect/", help="Path to trained models directory")
    parser.add_argument("--video", default="../provided_materials/tarefas_cima_3_parts.mp4", help="Path to input video")
    parser.add_argument("--display", default=True, help="Display predictions")
    args = parser.parse_args()

    model_name = get_latest_model(args.models_dir)
    model_path = f"{args.models_dir}/{model_name}/weights/best.pt"

    run_yolo_fsm_inference(model_path, args.video, display_predictions=args.display)
