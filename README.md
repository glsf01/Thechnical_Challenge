# Smart Factory Task Recognition – README

## Challenge
The objective of this project is to develop a system capable of detecting and classifying distinct moments in a manual or semi‑automated industrial operation, contributing to task recognition and performance analysis in a smart factory context.

The aim to automatically recognize four specific moments within each operation:
1. **PickUp** – when the operator picks up the piece.  
2. **ProbePass** – when the probe passes through the piece.  
3. **Marking** – when the operator makes a scratch/mark on the piece.  
4. **Place** – when the operator places the piece in the box.  

From these events, the system will compute:
- Average duration of a complete operation.  
- Percentage of operations where the probe passes over the pieces.  
- Percentage of operations where markings are made.  
- Total number of operations performed.  

The video data comes from a top‑down camera, where only the operator’s hands, the workpieces, and tools (probe and marker) are visible.

---

## Approach
This approach combines object detection, tracking, and temporal logic:

- **Object detection**: Train a YOLO‑based model (YOLO11n) to detect the relevant classes (`hand`, `piece`, `marker`, `probe`).
- **Event recognition**: Define a finite state machine (FSM) that interprets detection and tracking patterns to infer the four moments.  
- **Metrics**: Log timestamps of events to compute operation duration, percentages, and totals.

---

## Data Preparation with Roboflow for training
- The source video is split into frames at **5 frames per second (fps)** using Roboflow.  
- This sampling rate balances annotation effort with sufficient temporal resolution to capture hand/tool interactions.  
- Each frame is annotated with bounding boxes for the following classes:  
  - `hand`  
  - `piece`  
  - `marker`  
  - `probe`  

---

## Annotation Protocol
To ensure high‑quality, consistent annotations, practices from literature are followed:

1. **Label every instance** of the defined classes, even if partially visible or overlapping, with max of 15 annotations [1], [2].  
2. **Occluded objects**: Draw bounding boxes as if the object were fully visible (full extent), not just around the visible fragment. This improves consistency in box size and helps the model learn robust features under occlusion [1]–[3].  
3. **Overlapping subjects**: Each object gets its own bounding box, even if boxes overlap heavily (e.g., hand holding a probe) [1].  
4. **Stacked pieces**: Label each piece individually. If a piece is almost mostly hidden (e.g., less than 50% visible), it may be skipped to avoid noisy boxes [2].  
5. **Consistency over perfection**: Apply the same rules across the dataset. Consistency is more important than pixel‑perfect boxes [1], [3].  

---

## Implementation of Finite State Machine (FSM)
The Finite State Machine (FSM) is the core of the event recognition logic. It enforces the strict sequence of actions:

1. **PickUp**: Triggered when either hand overlaps with a piece.
2. **ProbePass**: Triggered when the probe overlaps with the piece. 
3. **Marking**: Triggered when the marker overlaps with the piece.
4. **Place**: Triggered when only one hand remains visible.

The FSM uses Intersection Over Union (IoU) to identify a possible event and a cool‑down mechanism to suppress duplicate events across consecutive frames and ensures that counters (probe_passes, markings) only increment when a new event is emitted. This prevents flooding of events when a tool remains overlapping the piece for multiple frames.

---

## How to Run:

1. Clone the repository and install dependencies
git clone https://github.com/glsf01/Thechnical_Challenge.git
pip install -r requirements.txt

2. Run object detection and visualization on a video at:
python scripts/model_inference.py

## References
[1] G. Ghanmi, “How to Label People in the Wild for Object Detection Tasks,” *Medium*, 2021. [Online]. Available: https://ghofrane-ghanmi01.medium.com/how-to-label-people-in-the-wild-for-object-detection-tasks-fb93ddc596d3  

[2] Esri, “Tips for Labeling Images for Object Detection Models,” *ArcGIS Blog*, 2023. [Online]. Available: https://www.esri.com/arcgis-blog/products/arcgis-pro/geoai/tips-for-labeling-images-for-object-detection-models  

[3] X. Ma, et al., “The Effect of Improving Annotation Quality on Object Detection Datasets,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2022, pp. 1–10.  

[4] Y. Wu, et al., “Robustness of Deep Learning Methods for Occluded Object Detection,” Univ. of Nottingham, Tech. Rep., 2021.  

[5] T. Wang, et al., “Robust Object Detection Under Occlusion With Context‑Aware CompositionalNets,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 12645–12654.
