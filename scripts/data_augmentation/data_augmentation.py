import os
import cv2
import yaml
import albumentations as A

# ------------------------------------------------------------
# Define augmentation pipeline
# ------------------------------------------------------------
def get_augmentation_pipeline():
    """
    Returns an Albumentations Compose object with the desired augmentations.
    The bbox_params ensure YOLO-format bounding boxes are updated automatically.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# ------------------------------------------------------------
# Load dataset paths from YAML
# ------------------------------------------------------------
def load_dataset_paths(yaml_path: str):
    """
    Loads dataset configuration from a YOLO-style YAML file.
    Returns the training image directory path.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data['train']


# ------------------------------------------------------------
# Load YOLO-format labels
# ------------------------------------------------------------
def load_labels(label_path: str):
    """
    Loads YOLO-format labels from a .txt file.
    Returns bounding boxes (list of [x, y, w, h]) and class labels.
    """
    bboxes, class_labels = [], []
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            bboxes.append([x, y, bw, bh])
            class_labels.append(int(cls))
    return bboxes, class_labels


# ------------------------------------------------------------
# Save augmented labels
# ------------------------------------------------------------
def save_labels(label_path: str, bboxes, class_labels):
    """
    Saves YOLO-format labels to a .txt file.
    """
    with open(label_path, "w") as f:
        for cls, (x, y, bw, bh) in zip(class_labels, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")


# ------------------------------------------------------------
# Process a single image + label pair
# ------------------------------------------------------------
def process_image(img_path: str, label_path: str, out_dir: str, transform):
    """
    Loads an image and its labels, applies augmentation, and saves the results.
    """
    image = cv2.imread(img_path)
    bboxes, class_labels = load_labels(label_path)

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_img, aug_boxes, aug_labels = augmented['image'], augmented['bboxes'], augmented['class_labels']

    # Save augmented image
    out_img_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_img_path, aug_img)

    # Save updated labels
    out_label_path = os.path.join(out_dir, os.path.basename(label_path))
    save_labels(out_label_path, aug_boxes, aug_labels)


# ------------------------------------------------------------
# Main augmentation loop
# ------------------------------------------------------------
def augment_dataset(yaml_path: str, out_dir: str):
    """
    Main function to augment a dataset:
    - Loads dataset paths from YAML
    - Iterates over all label files
    - Applies augmentations and saves results
    """
    img_dir = load_dataset_paths(yaml_path)
    os.makedirs(out_dir, exist_ok=True)
    transform = get_augmentation_pipeline()

    for label_file in [f for f in os.listdir(img_dir) if f.endswith(".txt")]:
        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(img_dir, label_file)

        if os.path.exists(img_path):
            process_image(img_path, label_path, out_dir, transform)


# ------------------------------------------------------------
# Run script
# ------------------------------------------------------------
if __name__ == "__main__":
    augment_dataset("data.yaml", "augmented")
