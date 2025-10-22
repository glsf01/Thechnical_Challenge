from ultralytics import YOLO

# TODO
#  1. Check dataset format that yolo uses
#   1.2 Maybe use roboflow as LabelStudio doesnt export in the YOLO format
#   label studio reference https://labelstud.io/guide/export
#   ultralitics train config https://docs.ultralytics.com/modes/train/#train-settings
# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model with custom configuration
model.train(data="", cfg="custom_data_augmentation.yaml")
