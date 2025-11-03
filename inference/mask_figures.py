import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

##### CONFIGURATION #####

# paths
OUTPUT_DIR = Path("outputs/masked_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

##### PARSE ARGUMENTS #####

if len(sys.argv) < 2:
    print("ERROR: No input path provided")
    print("Usage: python mask_figures_combined.py <detections_directory>")
    sys.exit(1)

input_path = Path(sys.argv[1])

##### CHECK INPUT #####

if not input_path.exists():
    print(f"ERROR: Path does not exist: {input_path}")
    sys.exit(1)

if not input_path.is_dir():
    print(f"ERROR: Path must be a directory: {input_path}")
    sys.exit(1)

##### COLLECT JSON FILES #####

json_files = list(input_path.glob("*_detections.json"))

if not json_files:
    print(f"ERROR: No detection JSON files found in: {input_path}")
    print("Run detect_figures.py first")
    sys.exit(1)

print(f"Found {len(json_files)} detection file(s)")
print("Reading bounding boxes from JSON")

##### MASKING FUNCTION #####

def create_masked_image_combined(original_img_path: Path, detections: list, image_size: dict) -> np.ndarray:
    # load original image
    original_img = Image.open(original_img_path)
    original_array = np.array(original_img)
    
    # create white canvas
    height, width = image_size["height"], image_size["width"]
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # place all figures on white canvas
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1 = bbox["x1"], bbox["y1"]
        x2, y2 = bbox["x2"], bbox["y2"]
        figure = original_array[y1:y2, x1:x2]
        
        # place this figure on white canvas at original position
        white_canvas[y1:y2, x1:x2] = figure
    
    return white_canvas

##### PROCESS JSON FILES #####

total_pages = 0
total_figures = 0

for json_file in sorted(json_files):
    print(f"Processing: {json_file.name}")
    
    # load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # extract info
    image_path = Path(data["image_path"])
    image_size = data["image_size"]
    num_figures = data["num_figures"]
    detections = data["detections"]
    
    if not image_path.exists():
        print(f"WARNING: Original image not found: {image_path}")
        continue
    
    print(f"  Figures: {num_figures}")
    
    # create one masked image with all figures
    masked_img = create_masked_image_combined(image_path, detections, image_size)
    
    # save masked image
    output_filename = f"{image_path.stem}_masked.png"
    output_path = OUTPUT_DIR / output_filename
    
    masked_pil = Image.fromarray(masked_img)
    masked_pil.save(output_path)
    
    print(f"  Saved: {output_filename}")
    
    total_figures += num_figures
    total_pages += 1

##### SUMMARY #####

print(f"\nPages processed: {total_pages}/{len(json_files)}")
print(f"Total figures: {total_figures}")
print(f"Masked images created: {total_pages}")
print(f"Output: {OUTPUT_DIR}")
