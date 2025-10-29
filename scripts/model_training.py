from ultralytics import YOLO


if __name__ == '__main__':
    # Load the YOLO11 model
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(
        cfg="../config/model_training_config.yaml",
    )

    # Export the model to ONNX format https://docs.ultralytics.com/integrations/onnx/#usage
    # model.export(format="onnx")  # creates 'yolo11n.onnx'
