import cv2 as cv

from ultralytics import YOLO


# REFERENCES:
# https://docs.ultralytics.com/modes/predict/#thread-safe-inference
# https://docs.opencv.org/4.12.0/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b

# Load the YOLO model
model_name = "hands_exp6"
model_path = f"./runs/detect/{model_name}/weights/best.pt"
model = YOLO(model_path)  # load a custom model

# Open the video file
video_path = "../../provided_materials/tarefas_cima.mp4"
cap = cv.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv.namedWindow("YOLO Inference", cv.WINDOW_NORMAL)
        cv.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()
