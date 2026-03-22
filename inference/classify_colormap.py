"""
JetFighter MLP Inference
-> classifies figures using trained Histogram-MLP model.

Classes:
    0: rainbow_gradient (problematic colormaps like jet, turbo)
    1: safe_gradient    (accessible colormaps like viridis, plasma)
    2: discrete         (discrete colors, no gradients)

Usage:
    python inference/classify_histogram.py <image> <detections.json> --visualize
    python inference/classify_histogram.py <image> <detections.json> --model models/histogram_classifier.pth
"""

import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser


### CONFIGURATION

OUTPUT_DIR = Path("outputs/histogram_classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = Path("models/histogram_classifier.pth")

# class mapping
CLASS_NAMES = {
    0: "rainbow_gradient",
    1: "safe_gradient",
    2: "discrete"
}

# colors for visualization (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),      # red for rainbow_gradient
    1: (0, 255, 0),      # green for safe_gradient
    2: (0, 165, 255)     # orange for discrete
}


class ColorClassifier(nn.Module):
    """
    Simple MLP (Multi-Layer Perceptron) for histogram classification.
    """
    
    def __init__(self, input_dim=512, num_classes=3, hidden_dim=256, dropout=0.3):
        super(ColorClassifier, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


def build_parser():
    parser = ArgumentParser(description='Classify figures with Histogram MLP model')
    
    parser.add_argument('image',
                        help='Input image (PDF page with multiple figures)')
    
    parser.add_argument('detections',
                        help='JSON file with bounding boxes (from outputs/detections/)')
    
    parser.add_argument('--model',
                        dest='model_path',
                        help=f'Path to trained model (default: {DEFAULT_MODEL})',
                        default=str(DEFAULT_MODEL))
    
    parser.add_argument('--visualize', '-v',
                        dest='visualize',
                        action='store_true',
                        help='Save annotated visualization')
    
    parser.add_argument('--confidence', '-c',
                        dest='conf_threshold',
                        type=float,
                        help='Confidence threshold for display (default: 0.0)',
                        default=0.0)
    
    return parser


def extract_histogram_features(img_array, bins=8, resize=600):
    # convert to PIL for resize
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # resize (if desired)
    if resize and resize > 0:
        img_pil = img_pil.resize((resize, resize))
    
    arr = np.array(img_pil)
    
    # 3D color histogram
    hist, edges = np.histogramdd(
        arr.reshape(-1, 3), 
        bins=(bins, bins, bins), 
        range=((0, 256), (0, 256), (0, 256))
    )
    
    # normalization
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum
    
    # flatten to 1D vector
    features = torch.FloatTensor(hist.flatten())
    
    return features


def load_model(model_path):
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    
    # load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # extract config
    config = checkpoint.get('config', {})
    bins = config.get('bins', 8)
    input_dim = config.get('input_dim', bins ** 3)
    resize = config.get('resize', 600)
    
    print(f"Model config:")
    print(f"  Bins: {bins}x{bins}x{bins} = {input_dim} features")
    print(f"  Resize: {resize}")
    
    # create model
    model = ColorClassifier(input_dim=input_dim, num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    
    return model, config


def classify_image_with_json(image_path, json_path, model, config, visualize=False, conf_threshold=0.0):
    # load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image {image_path}")
        return
    
    # load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    detections = data.get("detections", [])
    output_img = img.copy()
    
    results_summary = {
        "rainbow_gradient": 0,
        "safe_gradient": 0,
        "discrete": 0
    }
    
    print(f"Analyzing: {data.get('image_name', image_path.name)}")
    print(f"Detected figures: {len(detections)}\n")
    
    # extract config
    bins = config.get('bins', 8)
    resize = config.get('resize', 600)
    
    # classify each detection
    classification_results = []
    
    with torch.no_grad():
        for det in detections:
            fid = det.get("figure_id", "?")
            bbox = det["bbox"]
            
            # extract coordinates
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            # ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # extract crop
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"Figure {fid}: Empty crop, skipping")
                continue
            
            # extract histogram features
            features = extract_histogram_features(crop, bins=bins, resize=resize)
            features = features.unsqueeze(0)  # batch dim
            
            # predict
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            
            # top-1 prediction
            confidence, predicted_class = torch.max(probs, dim=1)
            predicted_class = int(predicted_class[0])
            confidence = float(confidence[0])
            
            class_name = CLASS_NAMES.get(predicted_class, "unknown")
            color = CLASS_COLORS.get(predicted_class, (128, 128, 128))
            
            # update summary
            results_summary[class_name] += 1
            
            # store result for JSON export
            classification_results.append({
                "figure_id": fid,
                "bbox": bbox,
                "class": class_name,
                "class_id": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    CLASS_NAMES[i]: float(probs[0][i])
                    for i in range(len(CLASS_NAMES))
                }
            })
            
            # console output
            print(f"Figure {fid}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.2%}")
            
            # probabilities
            for i in range(len(CLASS_NAMES)):
                name = CLASS_NAMES.get(i, "unknown")
                prob = float(probs[0][i])
                print(f"    {name}: {prob:.2%}")
            print()
            
            # visualization
            if visualize:
                # draw bounding box
                thickness = 4
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)
                
                # prepare label
                label = f"{class_name} ({confidence:.0%})"
                
                # label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_img, (x1, y1 - 25), (x1 + tw + 10, y1), color, -1)
                
                # label text
                cv2.putText(output_img, label, (x1 + 5, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    ### SUMMARY
    print(f"SUMMARY")
    print(f"rainbow_gradient: {results_summary['rainbow_gradient']}")
    print(f"safe_gradient:    {results_summary['safe_gradient']}")
    print(f"discrete:         {results_summary['discrete']}")
    
    # save JSON results
    json_output = {
        "image_name": data.get('image_name', Path(image_path).name),
        "model": str(Path(model_path).name) if 'model_path' in globals() else "unknown",
        "classifications": classification_results,
        "summary": results_summary
    }
    
    json_output_path = OUTPUT_DIR / f"{Path(image_path).stem}_classifications.json"
    with open(json_output_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Classification JSON saved: {json_output_path}\n")
    
    # save visualization
    if visualize:
        output_path = OUTPUT_DIR / f"{Path(image_path).stem}_classified.png"
        cv2.imwrite(str(output_path), output_img)
        print(f"Visualization saved: {output_path}\n")


def process_folder(folder_path, model, config, visualize=False, conf_threshold=0.0):
    folder_path = Path(folder_path)
    
    # find all JSON files
    json_files = list(folder_path.glob("*.json"))
    
    if len(json_files) == 0:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON files in {folder_path}\n")
    
    for json_file in sorted(json_files):
        # find corresponding image
        image_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            candidate = folder_path / f"{json_file.stem}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file is None:
            print(f"WARNING: No image found for {json_file.name}, skipping\n")
            continue
        
        # process the pair
        classify_image_with_json(image_file, json_file, model, config, visualize, conf_threshold)


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("Colormap Classification")
    
    # load model
    try:
        model, config = load_model(args.model_path)
    except Exception as e:
        print(f"ERROR: Could not load model - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # process input
    input_path = Path(args.image)
    detections_path = Path(args.detections)
    
    if not input_path.exists():
        print(f"ERROR: Image not found: {input_path}")
        return 1
    
    if input_path.is_file():
        # single file mode
        if not detections_path.exists():
            print(f"ERROR: Detections file not found: {detections_path}")
            return 1
        
        classify_image_with_json(
            input_path, 
            detections_path, 
            model,
            config,
            visualize=args.visualize,
            conf_threshold=args.conf_threshold
        )
    
    elif input_path.is_dir():
        # folder mode - search for image+JSON pairs
        process_folder(
            input_path, 
            model,
            config,
            visualize=args.visualize,
            conf_threshold=args.conf_threshold
        )
    
    else:
        print(f"ERROR: Invalid input path: {input_path}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
