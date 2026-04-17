"""
JetFighter — Combined Analysis Pipeline
Runs detect_figures -> classify_histogram -> contrast_analysis in sequence.

Categories:
    rainbow_gradient      -> RED   (problematic)
    safe_gradient         -> GREEN (safe)
    accessible_discrete   -> GREEN (safe)
    problematic_discrete  -> RED   (problematic)
"""

import os
import warnings
import cv2
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

### Optional imports
try:
    import torch
    import torch.nn as nn
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            DEVICE = "cuda"
        except RuntimeError:
            print("WARNING: CUDA available but kernel incompatible – falling back to CPU")
            DEVICE = "cpu"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


### CONSTANTS

# Detection (detect_figures.py)
DETECTION_CONFIDENCE = 0.40

# Classification (classify_histogram.py)
CLASS_NAMES = {0: "rainbow_gradient", 1: "safe_gradient", 2: "discrete"}

# Contrast analysis (contrast_analysis.py)
MIN_L              = 13
MAX_L              = 250
MIN_K              = 2
MAX_K              = 8
MAX_PIXELS         = 30_000
MIN_DATA_PIXELS    = 100
MIN_CLUSTER_FRAC   = 0.01
MIN_DELTA_E        = 35.0
MAX_DELTA_L        = 10.0

# Category
CATEGORY_INFO = {
    "rainbow_gradient":     {"status": "red",   "reason": "Rainbow / Jet colormap detected"},
    "safe_gradient":        {"status": "green", "reason": "Safe gradient colormap (e.g. viridis)"},
    "accessible_discrete":  {"status": "green", "reason": "Discrete plot with sufficient contrast"},
    "problematic_discrete": {"status": "red",   "reason": "Poor grayscale contrast in discrete plot"},
}

# Histogram Classifier Settings
HISTOGRAM_BINS = 8  # 8x8x8 = 512 features
try:
    HISTOGRAM_RESIZE = max(128, int(os.environ.get("HISTOGRAM_RESIZE", "512")))
except ValueError:
    HISTOGRAM_RESIZE = 512

# Spatial pre-filter for edge detection before histogram classification (to catch highly discrete figures liek ishihara plates)
SPATIAL_PREFILTER_SIZE = 256
SPATIAL_PREFILTER_GRID = 8
SPATIAL_PREFILTER_BORDER_FRAC = 0.08
SPATIAL_CANNY_LOW = 80
SPATIAL_CANNY_HIGH = 180
SPATIAL_EDGE_DENSITY_THRESHOLD = 0.085
SPATIAL_ACTIVE_TILE_DENSITY = 0.04
SPATIAL_ACTIVE_TILE_RATIO_THRESHOLD = 0.45
SPATIAL_VERY_HIGH_EDGE_DENSITY = 0.14


### HISTOGRAM CLASSIFIER MODEL

class ColorClassifier(nn.Module):
    
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


def extract_histogram(crop_rgb, bins=8, resize=512):
    if resize and resize > 0:
        h, w = crop_rgb.shape[:2]
        if h != resize or w != resize:
            interp = cv2.INTER_AREA if (h > resize or w > resize) else cv2.INTER_LINEAR
            crop_rgb = cv2.resize(crop_rgb, (resize, resize), interpolation=interp)

    hist = cv2.calcHist(
        [crop_rgb],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256],
    ).astype(np.float32)

    # Remove white bin (same convention as training)
    hist[bins - 1, bins - 1, bins - 1] = 0.0

    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum

    return hist.flatten()


def _spatial_prefilter_confidence(edge_density, active_tile_ratio):
    # Maps spatial filter strength to a bounded confidence for transparency
    density_norm = min(edge_density / max(SPATIAL_VERY_HIGH_EDGE_DENSITY, 1e-6), 1.0)
    spread_norm = min(active_tile_ratio / max(SPATIAL_ACTIVE_TILE_RATIO_THRESHOLD, 1e-6), 1.0)
    conf = 0.72 + 0.14 * density_norm + 0.14 * spread_norm
    return float(np.clip(conf, 0.72, 0.96))


def edge_detection(crop_bgr):
    # Spatial pre-filter for edge detection before histogram classification (to catch highly discrete figures liek ishihara plates)
    if crop_bgr is None or crop_bgr.size == 0:
        return False, 0.0, 0.0

    small = cv2.resize(crop_bgr,
                       (SPATIAL_PREFILTER_SIZE, SPATIAL_PREFILTER_SIZE),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, SPATIAL_CANNY_LOW, SPATIAL_CANNY_HIGH, L2gradient=True)

    edge_mask = edges > 0

    margin = int(SPATIAL_PREFILTER_SIZE * SPATIAL_PREFILTER_BORDER_FRAC)
    if margin > 0:
        inner = edge_mask[margin:-margin, margin:-margin]
        if inner.size == 0:
            inner = edge_mask
    else:
        inner = edge_mask

    edge_density = float(np.mean(inner))

    h, w = inner.shape
    active_tiles = 0
    total_tiles = 0
    for gy in range(SPATIAL_PREFILTER_GRID):
        y0 = (gy * h) // SPATIAL_PREFILTER_GRID
        y1 = ((gy + 1) * h) // SPATIAL_PREFILTER_GRID
        for gx in range(SPATIAL_PREFILTER_GRID):
            x0 = (gx * w) // SPATIAL_PREFILTER_GRID
            x1 = ((gx + 1) * w) // SPATIAL_PREFILTER_GRID
            tile = inner[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            total_tiles += 1
            if float(np.mean(tile)) >= SPATIAL_ACTIVE_TILE_DENSITY:
                active_tiles += 1

    active_tile_ratio = active_tiles / max(total_tiles, 1)

    is_discrete = (
        edge_density >= SPATIAL_VERY_HIGH_EDGE_DENSITY or
        (
            edge_density >= SPATIAL_EDGE_DENSITY_THRESHOLD and
            active_tile_ratio >= SPATIAL_ACTIVE_TILE_RATIO_THRESHOLD
        )
    )

    return is_discrete, edge_density, active_tile_ratio


### CONTRAST ANALYSIS HELPERS

def _cv_lab_to_cie(L, a, b):
    return L * (100.0 / 255.0), a - 128.0, b - 128.0

def _delta_e(lab1, lab2):
    L1, a1, b1 = _cv_lab_to_cie(lab1[0], lab1[1], lab1[2])
    L2, a2, b2 = _cv_lab_to_cie(lab2[0], lab2[1], lab2[2])
    return float(np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2))

def _delta_l(lab1, lab2):
    return float(abs(lab1[0] - lab2[0]) * (100.0 / 255.0))

def _lab_to_bgr(c):
    p = np.clip(c, 0, 255).reshape(1, 1, 3).astype(np.uint8)
    return cv2.cvtColor(p, cv2.COLOR_LAB2BGR)[0, 0]

def _gray709(bgr):
    return 0.0722 * bgr[0] + 0.7152 * bgr[1] + 0.2126 * bgr[2]

def _rel_lum(r, g, b):
    def _lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

def _wcag_cr(l1, l2):
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _best_k(data):
    n = len(data)
    lo, hi = MIN_K, min(MAX_K, n - 1)
    if n < lo + 1 or hi < lo:
        return lo
    best_k, best_s = lo, -1.0
    for k in range(lo, hi + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            lbl = km.fit_predict(data)
            if len(np.unique(lbl)) < 2:
                continue
            s = silhouette_score(data, lbl, sample_size=min(5000, n))
            if s > best_s:
                best_s, best_k = s, k
        except Exception:
            continue
    return best_k


def analyze_contrast_loss(crop_bgr,
                          min_delta_e=MIN_DELTA_E,
                          max_delta_l=MAX_DELTA_L):

    if not _HAS_SKLEARN:
        return dict(verdict="ACCESSIBLE", score=0.0, n_colors=0,
                    problematic_pairs=[], cluster_info=[],
                    reason="scikit-learn not available.")

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float64)
    mask = (L >= MIN_L) & (L <= MAX_L)
    n_data = int(np.sum(mask))

    _empty = dict(verdict="ACCESSIBLE", score=0.0, n_colors=0,
                  problematic_pairs=[], cluster_info=[])

    if n_data < MIN_DATA_PIXELS:
        return {**_empty, "reason": f"Too few data pixels ({n_data})."}

    data_lab = lab[mask].astype(np.float64)

    if len(data_lab) > MAX_PIXELS:
        idx = np.random.RandomState(42).choice(len(data_lab), MAX_PIXELS, replace=False)
        data_lab = data_lab[idx]

    k = _best_k(data_lab)
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(data_lab)
    centroids = km.cluster_centers_
    unique, counts = np.unique(labels, return_counts=True)
    fracs = counts / len(labels)
    sig = unique[fracs >= MIN_CLUSTER_FRAC]

    if len(sig) < 2:
        return {**_empty, "n_colors": int(len(sig)),
                "reason": "Fewer than 2 significant clusters."}

    # Build cluster info
    cluster_info = []
    for ci in sig:
        c_bgr = _lab_to_bgr(centroids[ci])
        cluster_info.append(dict(
            cluster_id=int(ci),
            fraction=round(float(fracs[ci]), 4),
            centroid_rgb=[int(c_bgr[2]), int(c_bgr[1]), int(c_bgr[0])],
            L_star=round(float(centroids[ci][0] * 100.0 / 255.0), 2),
            grayscale=round(float(_gray709(c_bgr.astype(np.float64))), 1),
        ))

    # Pairwise analysis
    prob_pairs = []
    total_pairs = 0

    for i in range(len(sig)):
        for j in range(i + 1, len(sig)):
            total_pairs += 1
            ci_lab, cj_lab = centroids[sig[i]], centroids[sig[j]]
            de = _delta_e(ci_lab, cj_lab)
            dl = _delta_l(ci_lab, cj_lab)

            if de >= min_delta_e and dl < max_delta_l:
                ci_bgr = _lab_to_bgr(ci_lab).astype(np.float64)
                cj_bgr = _lab_to_bgr(cj_lab).astype(np.float64)
                gi, gj = _gray709(ci_bgr), _gray709(cj_bgr)
                li = _rel_lum(ci_bgr[2], ci_bgr[1], ci_bgr[0])
                lj = _rel_lum(cj_bgr[2], cj_bgr[1], cj_bgr[0])

                prob_pairs.append(dict(
                    color_a_rgb=cluster_info[i]["centroid_rgb"],
                    color_b_rgb=cluster_info[j]["centroid_rgb"],
                    delta_e=round(de, 2),
                    delta_l_star=round(dl, 2),
                    grayscale_a=round(gi, 1),
                    grayscale_b=round(gj, 1),
                    grayscale_diff=round(abs(gi - gj), 1),
                    wcag_contrast_ratio=round(_wcag_cr(li, lj), 2),
                ))

    if prob_pairs:
        score = round(len(prob_pairs) / max(total_pairs, 1), 3)
        descs = [f"RGB{tuple(p['color_a_rgb'])} vs RGB{tuple(p['color_b_rgb'])} "
                 f"(ΔE={p['delta_e']}, ΔL*={p['delta_l_star']})"
                 for p in prob_pairs]
        reason = (f"{len(prob_pairs)} colour pair(s) indistinguishable in "
                  f"grayscale: " + "; ".join(descs))
        verdict = "PROBLEMATIC"
    else:
        score = 0.0
        verdict = "ACCESSIBLE"
        reason = f"All {total_pairs} pair(s) have sufficient luminance contrast."

    return dict(verdict=verdict, score=score, n_colors=len(sig),
                problematic_pairs=prob_pairs, cluster_info=cluster_info,
                reason=reason)


### PIPELINE CLASS

class JetFighterPipeline:

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.detector = None
        self.classifier = None
        self.histogram_bins = HISTOGRAM_BINS
        self._load_models()

    def _load_models(self):
        if not _HAS_YOLO:
            print("[Pipeline] ultralytics not installed")
            return

        # Detector
        det_path = self.models_dir / "detector2_training" / "yolo_detector" / "weights" / "best.pt"
        if det_path.exists():
            self.detector = YOLO(str(det_path))
            self.detector.to(DEVICE)
            print(f"[Pipeline] Detector loaded on {DEVICE.upper()}: {det_path}")
        else:
            print(f"[Pipeline] WARNING: detector not found at {det_path}")

        # Classifier (histogram model)
        cls_path = self.models_dir / "histogram_training3" / "run_20260215_165650" / "weights" / "best.pth"
        if cls_path.exists():
            try:
                input_dim = self.histogram_bins ** 3
                self.classifier = ColorClassifier(input_dim=input_dim, num_classes=3)
                
                # Load checkpoint (contains model_state_dict, config, metrics)
                checkpoint = torch.load(str(cls_path), map_location=DEVICE)
                
                # Extract the actual state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                self.classifier.load_state_dict(state_dict)
                self.classifier.to(DEVICE)
                self.classifier.eval()
                print(f"[Pipeline] Histogram classifier loaded on {DEVICE.upper()}: {cls_path}")
            except Exception as e:
                print(f"[Pipeline] ERROR loading classifier: {e}")
                self.classifier = None
        else:
            print(f"[Pipeline] WARNING: classifier model not found at {cls_path}")

    # Step 1: Figure Detection
    def _detect_figures(self, image_path):
        """Returns list[dict] with figure_id, bbox, confidence."""
        if self.detector is None:
            print("[Pipeline] ERROR: No detector model loaded.")
            return []

        results = self.detector(str(image_path), device=DEVICE, verbose=False)
        boxes = results[0].boxes
        filtered = [b for b in boxes if float(b.conf[0]) >= DETECTION_CONFIDENCE]

        dets = []
        for i, box in enumerate(filtered):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            dets.append(dict(
                figure_id=i + 1,
                bbox=dict(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=round(float(box.conf[0].cpu().numpy()), 4),
            ))
        return dets

    # Step 2: Classification
    def _classify(self, crop_bgr):
        is_discrete, edge_density, active_tile_ratio = edge_detection(crop_bgr)
        if is_discrete:
            conf = round(_spatial_prefilter_confidence(edge_density, active_tile_ratio), 4)
            residual = max(0.0, 1.0 - conf)
            return "discrete", conf, {
                "rainbow_gradient": round(residual * 0.5, 4),
                "safe_gradient": round(residual * 0.5, 4),
                "discrete": conf,
            }, "spatial_prefilter"

        if self.classifier is None:
            print("[Pipeline] ERROR: No classifier model loaded.")
            return "unknown", 0.0, {}, "none"

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract histogram features
        hist_features = extract_histogram(crop_rgb, bins=self.histogram_bins, resize=HISTOGRAM_RESIZE)
        
        # Run inference
        with torch.no_grad():
            features_tensor = torch.FloatTensor(hist_features).unsqueeze(0).to(DEVICE)
            logits = self.classifier(features_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            cls_id = int(torch.argmax(probs).cpu())
            conf = float(probs[cls_id].cpu())
            name = CLASS_NAMES.get(cls_id, "unknown")
            
            prob_dict = {}
            for i, class_name in CLASS_NAMES.items():
                prob_dict[class_name] = round(float(probs[i].cpu()), 4)

        return name, conf, prob_dict, "mlp"

    # Full page pipeline
    def analyze_page(self, image_path):
        """Run detect -> classify -> contrast on one page image."""
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            return dict(figures=[], image_width=0, image_height=0, num_figures=0,
                        error=f"Could not load {image_path}")

        h, w = img.shape[:2]
        detections = self._detect_figures(image_path)

        figures = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
            x2, y2 = min(w, bbox["x2"]), min(h, bbox["y2"])

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cls_name, cls_conf, cls_probs, cls_source = self._classify(crop)

            contrast = None
            if cls_name == "discrete":
                contrast = analyze_contrast_loss(crop)
                category = ("problematic_discrete"
                            if contrast["verdict"] == "PROBLEMATIC"
                            else "accessible_discrete")
            else:
                category = cls_name

            info = CATEGORY_INFO[category]
            reason = info["reason"]
            if contrast and category == "problematic_discrete":
                reason = contrast["reason"]

            figures.append(dict(
                figure_id=det["figure_id"],
                bbox=dict(x1=x1, y1=y1, x2=x2, y2=y2),
                category=category,
                status=info["status"],
                reason=reason,
                classification=cls_name,
                classification_source=cls_source,
                classification_confidence=round(cls_conf, 4),
                classification_probabilities=cls_probs,
                detection_confidence=det["confidence"],
                contrast_details=contrast,
            ))

        return dict(
            figures=figures,
            image_width=w,
            image_height=h,
            num_figures=len(figures),
        )

    # Classify a single image (no detection, treat entire image as one figure)
    def analyze_image(self, image_path):
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            return dict(figures=[], image_width=0, image_height=0, num_figures=0,
                        error=f"Could not load {image_path}")

        h, w = img.shape[:2]

        # Classify the whole image as one figure
        cls_name, cls_conf, cls_probs, cls_source = self._classify(img)

        contrast = None
        if cls_name == "discrete":
            contrast = analyze_contrast_loss(img)
            category = ("problematic_discrete"
                        if contrast["verdict"] == "PROBLEMATIC"
                        else "accessible_discrete")
        else:
            category = cls_name

        info = CATEGORY_INFO[category]
        reason = info["reason"]
        if contrast and category == "problematic_discrete":
            reason = contrast["reason"]

        figure = dict(
            figure_id=1,
            bbox=dict(x1=0, y1=0, x2=w, y2=h),
            category=category,
            status=info["status"],
            reason=reason,
            classification=cls_name,
            classification_source=cls_source,
            classification_confidence=round(cls_conf, 4),
            classification_probabilities=cls_probs,
            detection_confidence=1.0,
            contrast_details=contrast,
        )

        return dict(
            figures=[figure],
            image_width=w,
            image_height=h,
            num_figures=1,
        )
