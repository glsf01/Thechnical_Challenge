import cv2 as cv
from ultralytics import YOLO
from aux import get_latest_model

def run_yolo_inference(model_path: str, video_path: str, window_name: str = "YOLO Inference"):
    """
    Run YOLO inference on a video and display annotated frames.
    :param model_path: Path to the YOLO model weights (.pt file).
    :param video_path: Path to the input video file.
    :param window_name: Name of the OpenCV display window.
    :return:
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

        # Display the annotated frame
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    trained_models = "../../runs/detect/"
    model_name = get_latest_model(trained_models)
    model_path = f"../../runs/detect/{model_name}/weights/best.pt"
    video_path = "../../provided_materials/tarefas_cima.mp4"

    run_yolo_inference(model_path, video_path)
