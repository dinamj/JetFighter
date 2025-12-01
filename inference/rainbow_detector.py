import sys
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

##### CONFIGURATION #####

# detect device (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paths
MODEL_PATH = Path("models/rainbow_detector.pt")
OUTPUT_DIR = Path("outputs/rainbow_detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# parameters
CONFIDENCE_THRESHOLD = 0.5

# visualization settings
BBOX_COLOR = (255, 0, 255)  # magenta for rainbow
BBOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

##### PARSE ARGUMENTS #####

if len(sys.argv) < 2:
    print("ERROR: No input path provided")
    print("Usage: python rainbow_detector.py <image_path> [--visualize]")
    sys.exit(1)

input_path = Path(sys.argv[1])
visualize = "--visualize" in sys.argv or "-v" in sys.argv

##### CHECK AND LOAD MODEL #####

if not MODEL_PATH.exists():
    print(f"ERROR: Rainbow detector model not found at {MODEL_PATH}")
    print("Train the model first: python training/train_rainbow_detector.py")
    sys.exit(1)

if not input_path.exists():
    print(f"ERROR: Image not found: {input_path}")
    sys.exit(1)

model = YOLO(str(MODEL_PATH))
model.to(DEVICE)
print(f"Rainbow detector loaded on {DEVICE.upper()}")

if visualize:
    print("Visualization enabled")

##### DETECTION #####

print(f"Analyzing: {input_path.name}")

# load image
img = cv2.imread(str(input_path))
if img is None:
    print(f"ERROR: Could not load image: {input_path}")
    sys.exit(1)

height, width = img.shape[:2]

# run detection
results = model.predict(
    source=str(input_path),
    conf=CONFIDENCE_THRESHOLD,
    device=DEVICE,
    verbose=False
)

# extract detections
has_rainbow = False
detections = []

if len(results) > 0 and results[0].boxes is not None:
    boxes = results[0].boxes
    if len(boxes) > 0:
        has_rainbow = True
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            confidence = float(box.conf[0].cpu().numpy())
            
            detection = {
                "detection_id": i + 1,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "confidence": confidence,
                "width": x2 - x1,
                "height": y2 - y1
            }
            detections.append(detection)

##### RESULTS #####

print("\n" + "="*50)
print("RAINBOW COLORMAP DETECTION RESULTS")
print("="*50)
print(f"Image: {input_path.name}")
print(f"Image size: {width}x{height}")
print(f"Rainbow detected: {'YES' if has_rainbow else 'NO'}")
print(f"Number of detections: {len(detections)}")

if detections:
    print("\nDetections:")
    for det in detections:
        print(f"  #{det['detection_id']}: Confidence {det['confidence']:.2f} | BBox: {det['bbox']}")
else:
    print("No rainbow colormaps detected")

##### VISUALIZATION #####

if visualize and has_rainbow:
    vis_img = img.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        conf = detection["confidence"]
        det_id = detection["detection_id"]
        
        # draw rectangle
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
        
        # draw label
        label = f"Rainbow {det_id}: {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        cv2.rectangle(
            vis_img,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            TEXT_BACKGROUND,
            -1
        )
        cv2.putText(
            vis_img,
            label,
            (x1 + 5, y1 - baseline - 5),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS
        )
    
    # save visualization
    vis_filename = f"{input_path.stem}_rainbow_detected.png"
    vis_path = OUTPUT_DIR / vis_filename
    cv2.imwrite(str(vis_path), vis_img)
    print(f"\nVisualization saved: {vis_path}")

print("="*50)
