import sys
import json
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

##### CONFIGURATION #####

# detect device (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paths
DETECTOR_MODEL = Path("models/detector.pt")
OUTPUT_DIR = Path("outputs/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# visualization settings
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

##### PARSE ARGUMENTS #####

if len(sys.argv) < 2:
    print("ERROR: No input path provided")
    print("Usage: python detect_figures.py <path> [--visualize]")
    sys.exit(1)

input_path = Path(sys.argv[1])
visualize = "--visualize" in sys.argv or "-v" in sys.argv

##### CHECK AND LOAD MODEL #####

if not DETECTOR_MODEL.exists():
    print(f"ERROR: Detector model not found at {DETECTOR_MODEL}")
    sys.exit(1)

if not input_path.exists():
    print(f"ERROR: Path does not exist: {input_path}")
    sys.exit(1)

detector = YOLO(str(DETECTOR_MODEL))
detector.to(DEVICE)
print(f"Detector loaded on {DEVICE.upper()}")

if visualize:
    print("Visualization enabled")

##### COLLECT IMAGES #####

image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
image_paths = []

if input_path.is_file():
    if input_path.suffix.lower() in image_extensions:
        image_paths.append(input_path)
    else:
        print(f"ERROR: Not a supported image file: {input_path}")
        sys.exit(1)
elif input_path.is_dir():
    for ext in image_extensions:
        image_paths.extend(input_path.glob(f"*{ext}"))
        image_paths.extend(input_path.glob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))
else:
    print(f"ERROR: Invalid path: {input_path}")
    sys.exit(1)

if not image_paths:
    print(f"ERROR: No images found in: {input_path}")
    sys.exit(1)

print(f"Found {len(image_paths)} image(s)")

##### DETECTION FUNCTION #####

def detect_and_save(image_path: Path, save_visualization: bool = False) -> dict:
    
    # load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    # run detection
    results = detector(str(image_path), device=DEVICE, verbose=False)
    boxes = results[0].boxes
    num_detections = len(boxes)
    
    # extract bounding box data
    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        confidence = float(box.conf[0].cpu().numpy())
        
        detection = {
            "figure_id": i + 1,
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
    
    # prepare output data
    output_data = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_size": {
            "width": width,
            "height": height
        },
        "timestamp": datetime.now().isoformat(),
        "num_figures": num_detections,
        "detections": detections
    }
    
    # save JSON
    json_filename = f"{image_path.stem}_detections.json"
    json_path = OUTPUT_DIR / json_filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # save visualization if requested
    if save_visualization and num_detections > 0:
        vis_img = img.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            conf = detection["confidence"]
            fig_id = detection["figure_id"]
            
            # draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
            
            # draw label
            label = f"Figure {fig_id}: {conf:.2f}"
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
        vis_filename = f"{image_path.stem}_visualized.png"
        vis_path = OUTPUT_DIR / vis_filename
        cv2.imwrite(str(vis_path), vis_img)
    
    return output_data

##### PROCESS IMAGES #####

total_figures = 0
processed_count = 0

for i, img_path in enumerate(image_paths, 1):
    print(f"[{i}/{len(image_paths)}] {img_path.name}")
    result = detect_and_save(img_path, save_visualization=visualize)
    
    if result:
        total_figures += result["num_figures"]
        processed_count += 1

##### SUMMARY #####

print(f"\nPages processed: {processed_count}/{len(image_paths)}")
print(f"Total figures: {total_figures}")
print(f"Output: {OUTPUT_DIR}")