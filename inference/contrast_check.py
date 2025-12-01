import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000

##### CONFIGURATION #####

DEUTERANOPIA_MATRIX = np.array([
    [0.457771, 0.731899, -0.184143],
    [0.226409, 0.731012, 0.040891],
    [-0.011553, 0.036433, 0.954719]
])

RATIO_THRESHOLD = 5.0
MIN_DISTANCE_THRESHOLD = 10.0
PIXEL_PERCENTAGE_THRESHOLD = 5.0
MAX_PALETTE_COLORS = 256

OUTPUT_DIR = Path("outputs/contrast_detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BBOX_COLOR = (255, 165, 0)
BBOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (255, 165, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

##### PARSE ARGUMENTS #####

if len(sys.argv) < 2:
    print("ERROR: No input path provided")
    print("Usage: python contrast_check.py <image_path> [--visualize]")
    sys.exit(1)

input_path = Path(sys.argv[1])
visualize = "--visualize" in sys.argv or "-v" in sys.argv

if not input_path.exists():
    print(f"ERROR: Image not found: {input_path}")
    sys.exit(1)

##### SIMULATION #####

def srgb_to_linear(rgb):
    rgb_norm = rgb.astype(np.float32) / 255.0
    return np.where(rgb_norm <= 0.04045, rgb_norm / 12.92, ((rgb_norm + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(rgb_linear):
    rgb_out = np.where(rgb_linear <= 0.0031308, rgb_linear * 12.92, 1.055 * (rgb_linear ** (1/2.4)) - 0.055)
    return (np.clip(rgb_out, 0, 1) * 255).astype(np.uint8)

def simulate_deuteranopia_array(rgb_array):
    rgb_linear = srgb_to_linear(rgb_array)
    simulated_linear = np.dot(rgb_linear, DEUTERANOPIA_MATRIX.T)
    simulated_linear = np.clip(simulated_linear, 0, 1)
    return linear_to_srgb(simulated_linear)

##### PALETTE EXTRACTION #####

def is_grayscale_color(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    
    if mx == 0:
        s = 0
    else:
        s = df / mx
    
    v = mx
    
    if v < 40:
        return True
    
    if v > 230 and s < 0.15:
        return True
    
    if s < 0.10:
        return True
    
    return False

def extract_palette_and_histogram(image_path):
    img = Image.open(image_path).convert('RGB')
    
    quantized = img.quantize(colors=MAX_PALETTE_COLORS, method=2)
    
    palette_flat = quantized.getpalette()
    palette_rgb = np.array(palette_flat[:MAX_PALETTE_COLORS*3]).reshape(-1, 3)
    
    color_counts = quantized.getcolors(MAX_PALETTE_COLORS)
    
    if color_counts is None:
        histogram = np.zeros(MAX_PALETTE_COLORS, dtype=int)
        n_colors = MAX_PALETTE_COLORS
    else:
        n_colors = len(color_counts)
        histogram = np.zeros(MAX_PALETTE_COLORS, dtype=int)
        for count, idx in color_counts:
            histogram[idx] = count
    
    palette_rgb = palette_rgb[:n_colors]
    histogram = histogram[:n_colors]
    
    non_gray_mask = np.array([not is_grayscale_color(rgb) for rgb in palette_rgb])
    
    return palette_rgb, histogram, non_gray_mask, img, quantized

##### DISTANCE CALCULATION #####

def calculate_distance_matrix(palette_rgb):
    n = len(palette_rgb)
    if n == 0:
        return np.array([])
    
    # convert to LAB color space
    lab_palette = rgb2lab(palette_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            delta_e = deltaE_ciede2000(lab_palette[i], lab_palette[j])
            dist_matrix[i, j] = delta_e
            dist_matrix[j, i] = delta_e
    
    return dist_matrix

##### DETECTION #####

print(f"Analyzing: {input_path.name}")

palette_rgb, histogram, non_gray_mask, original_img, quantized_img = extract_palette_and_histogram(str(input_path))

print(f"Extracted {len(palette_rgb)} colors ({original_img.width}x{original_img.height}px)")

palette_rgb_filtered = palette_rgb[non_gray_mask]
histogram_filtered = histogram[non_gray_mask]

print(f"Filtered to {len(palette_rgb_filtered)} non-grayscale colors")

if len(palette_rgb_filtered) < 2:
    print("No contrast loss detected")
    sys.exit(0)

palette_sim = simulate_deuteranopia_array(palette_rgb_filtered)

dist_orig = calculate_distance_matrix(palette_rgb_filtered)
dist_sim = calculate_distance_matrix(palette_sim)

ratio_matrix = np.divide(dist_orig, dist_sim + 0.1, where=(dist_sim + 0.1) != 0)

problematic_pairs = []
total_pixels = np.sum(histogram_filtered)
max_ratio = 0.0

for i in range(len(palette_rgb_filtered)):
    for j in range(i+1, len(palette_rgb_filtered)):
        current_ratio = ratio_matrix[i, j]
        if current_ratio > max_ratio:
            max_ratio = current_ratio
        
        if current_ratio > RATIO_THRESHOLD and dist_orig[i, j] > MIN_DISTANCE_THRESHOLD:
            pixel_count_i = histogram_filtered[i]
            pixel_count_j = histogram_filtered[j]
            combined_pixels = pixel_count_i + pixel_count_j
            
            problematic_pairs.append({
                'color_i': i,
                'color_j': j,
                'ratio': float(current_ratio),
                'dist_orig': float(dist_orig[i, j]),
                'dist_sim': float(dist_sim[i, j]),
                'pixels': int(combined_pixels),
                'rgb_i': tuple(palette_rgb_filtered[i]),
                'rgb_j': tuple(palette_rgb_filtered[j])
            })

affected_colors = set()
for pair in problematic_pairs:
    affected_colors.add(pair['color_i'])
    affected_colors.add(pair['color_j'])

affected_pixels = sum(histogram_filtered[c] for c in affected_colors)
affected_percentage = (affected_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

has_contrast_loss = affected_percentage > PIXEL_PERCENTAGE_THRESHOLD

##### RESULTS #####

print("\nCONTRAST LOSS DETECTION")
print(f"Image: {input_path.name}")
print(f"Palette Colors (non-gray): {len(palette_rgb_filtered)}")
print(f"Max Ratio: {max_ratio:.2f}")
print(f"Problematic Pairs: {len(problematic_pairs)}")
print(f"Affected Pixels: {affected_pixels} / {total_pixels}")
print(f"Percentage: {affected_percentage:.2f}%")
print(f"Contrast Loss: {'YES' if has_contrast_loss else 'NO'}")

##### VISUALIZATION #####

if visualize:
    original_img_pil = Image.open(str(input_path)).convert('RGB')
    img_array = np.array(original_img_pil)
    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    vis_img = img_bgr.copy()
    h, w = vis_img.shape[:2]
    
    if has_contrast_loss and problematic_pairs:
        quantized_array = np.array(quantized_img)
        
        # build set of problematic palette indices
        problematic_filtered_indices = set()
        for pair in problematic_pairs:
            problematic_filtered_indices.add(pair['color_i'])
            problematic_filtered_indices.add(pair['color_j'])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # map filtered indices back to original palette indices
        # and mark corresponding pixels in the mask
        for filtered_idx in problematic_filtered_indices:
            original_idx = np.where(non_gray_mask)[0][filtered_idx]
            pixel_mask = (quantized_array == original_idx)
            mask[pixel_mask] = 255
        
        # clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        problem_regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= 9000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                problem_regions.append({
                    "id": i + 1,
                    "bbox": {"x1": x, "y1": y, "x2": x + w_box, "y2": y + h_box},
                    "area": int(area)
                })
        
        for region in problem_regions:
            bbox = region["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            region_id = region["id"]
            area = region["area"]
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
            
            label = f"Region {region_id}: {area}px"
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
        
        print(f"\nProblem Regions: {len(problem_regions)}")
    
    vis_filename = f"{input_path.stem}_contrast_loss.png"
    vis_path = OUTPUT_DIR / vis_filename
    cv2.imwrite(str(vis_path), vis_img)
    print(f"\nVisualization saved: {vis_path}")
    
    img_sim_array = simulate_deuteranopia_array(img_array.reshape(-1, 3)).reshape(h, w, 3)
    img_sim_bgr = cv2.cvtColor(img_sim_array, cv2.COLOR_RGB2BGR)
    
    sim_filename = f"{input_path.stem}_deuteranopia_sim.png"
    sim_path = OUTPUT_DIR / sim_filename
    cv2.imwrite(str(sim_path), img_sim_bgr)
    print(f"Simulated image: {sim_path}")
