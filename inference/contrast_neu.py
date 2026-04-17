"""
JetFighter — Contrast Loss Analysis (Step 3)
-> Detects if distinct colors in discrete figures become indistinguishable
   in grayscale, making the figure inaccessible for colorblind readers.

Usage:
    python inference/contrast_analysis.py <image> <classifications.json> --visualize
    python inference/contrast_analysis.py <image> <detections.json> --analyze-all -v
    python inference/contrast_analysis.py <img_folder> <json_folder> --visualize
    python inference/contrast_analysis.py <image_or_folder> --visualize
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


### CONFIGURATION

OUTPUT_DIR = Path("outputs/contrast_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-processing (OpenCV LAB scale)
MIN_L = 13          # reject pure-black pixels
MAX_L = 250         # reject paper-white background

# Clustering
MIN_K = 2
MAX_K = 8
MAX_PIXELS = 30000
MIN_DATA_PIXELS = 100
MIN_CLUSTER_FRACTION = 0.01

# Contrast thresholds
MIN_DELTA_E = 35.0  # min ΔE to consider colors distinct
MAX_DELTA_L = 10.0  # max ΔL* — below this, grays are indistinguishable

# Visualization (BGR)
COLOR_PROBLEMATIC = (0, 0, 255)    # red
COLOR_ACCESSIBLE  = (0, 200, 0)    # green
BBOX_THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


### CLI

def build_parser():
    p = ArgumentParser(
        description="Contrast Loss Analysis for Discrete Figures (Step 3)")

    p.add_argument("image",
                   help="Input image file or folder of images")
    p.add_argument("detections", nargs="?", default=None,
                   help="Optional classification / detection JSON file or folder")

    p.add_argument("--visualize", "-v", dest="visualize",
                   action="store_true",
                   help="Save grayscale visualisation with annotated bboxes")
    p.add_argument("--analyze-all", dest="analyze_all",
                   action="store_true",
                   help="Analyse ALL figures regardless of class label "
                        "(use with detection-only JSONs)")

    p.add_argument("--delta-e", type=float, default=MIN_DELTA_E,
                   help=f"Min ΔE to consider colors distinct "
                        f"(default: {MIN_DELTA_E})")
    p.add_argument("--delta-l", type=float, default=MAX_DELTA_L,
                   help=f"Max ΔL* to flag as indistinguishable "
                        f"(default: {MAX_DELTA_L})")
    return p


### COLOR-SCIENCE HELPERS

def _opencv_lab_to_cie(L_cv, a_cv, b_cv):
    # OpenCV LAB -> CIE L*a*b*
    return L_cv * (100.0 / 255.0), a_cv - 128.0, b_cv - 128.0


def delta_e_ab(lab1_cv, lab2_cv):
    # Euclidean ΔE_ab in CIE L*a*b* given two OpenCV-LAB vectors
    L1, a1, b1 = _opencv_lab_to_cie(lab1_cv[0], lab1_cv[1], lab1_cv[2])
    L2, a2, b2 = _opencv_lab_to_cie(lab2_cv[0], lab2_cv[1], lab2_cv[2])
    return float(np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2))


def delta_l_star(lab1_cv, lab2_cv):
    # Absolute ΔL* in CIE scale from two OpenCV-LAB vectors
    return float(abs(lab1_cv[0] - lab2_cv[0]) * (100.0 / 255.0))


def _chroma_cv(a_cv, b_cv):
    # Chroma from OpenCV a, b channels
    return np.sqrt((a_cv - 128.0)**2 + (b_cv - 128.0)**2)


def _lab_centroid_to_bgr(centroid_lab):
    # Convert a single LAB centroid (float) -> BGR uint8 tuple
    lab = np.clip(centroid_lab, 0, 255).reshape(1, 1, 3).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr[0, 0]


def _grayscale_bt709(bgr):
    # BT.709 grayscale from BGR floats
    return 0.0722 * bgr[0] + 0.7152 * bgr[1] + 0.2126 * bgr[2]


def _srgb_relative_luminance(r, g, b):
    #WCAG relative luminance from sRGB 0-255
    def _lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _wcag_contrast_ratio(lum1, lum2):
    lighter, darker = max(lum1, lum2), min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


### CORE ANALYSIS

def _filter_data_pixels(lab_image, min_l, max_l):
    # mask out white/black pixels
    L = lab_image[:, :, 0].astype(np.float64)
    return (L >= min_l) & (L <= max_l)


def _best_k(data, min_k=MIN_K, max_k=MAX_K):
    # Pick K with highest silhouette score
    n = len(data)
    if n < min_k + 1:
        return min_k
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        return min_k

    best, best_s = min_k, -1.0
    for k in range(min_k, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42,
                        max_iter=300)
            labels = km.fit_predict(data)
            if len(np.unique(labels)) < 2:
                continue
            s = silhouette_score(data, labels,
                                 sample_size=min(5000, n))
            if s > best_s:
                best_s, best = s, k
        except Exception:
            continue
    return best


def analyze_contrast_loss(crop_bgr, min_delta_e=MIN_DELTA_E,
                          max_delta_l=MAX_DELTA_L):
    # Convert to LAB & filter
    lab_image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    mask = _filter_data_pixels(lab_image, MIN_L, MAX_L)
    n_data = int(np.sum(mask))

    _empty = dict(verdict="ACCESSIBLE", n_colors=0,
                  problematic_pairs=[], all_pairs=[])

    if n_data < MIN_DATA_PIXELS:
        return {**_empty,
                "reason": f"Too few data pixels ({n_data}) — "
                          f"figure likely does not rely on color coding."}

    data_lab = lab_image[mask].astype(np.float64)
    data_bgr = crop_bgr[mask]

    # Subsample for speed
    if len(data_lab) > MAX_PIXELS:
        idx = np.random.RandomState(42).choice(
            len(data_lab), MAX_PIXELS, replace=False)
        data_lab = data_lab[idx]
        data_bgr = data_bgr[idx]

    # Cluster
    k = _best_k(data_lab)
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(data_lab)
    centroids = km.cluster_centers_

    unique, counts = np.unique(labels, return_counts=True)
    fracs = counts / len(labels)

    sig = unique[fracs >= MIN_CLUSTER_FRACTION]

    if len(sig) < 2:
        return {**_empty, "n_colors": int(len(sig)),
                "reason": "Fewer than 2 significant color clusters — "
                          "no color contrast to evaluate."}

    # Build cluster info
    cluster_info = []
    for ci in sig:
        c_lab = centroids[ci]
        c_bgr = _lab_centroid_to_bgr(c_lab)
        L_star = float(c_lab[0] * (100.0 / 255.0))
        gray = float(_grayscale_bt709(c_bgr.astype(np.float64)))
        cluster_info.append(dict(
            cluster_id=int(ci),
            fraction=round(float(fracs[ci]), 4),
            centroid_rgb=[int(c_bgr[2]), int(c_bgr[1]), int(c_bgr[0])],
            centroid_bgr=c_bgr.tolist(),
            L_star=round(L_star, 2),
            grayscale=round(gray, 1),
        ))

    # Pair-wise analysis
    all_pairs, problematic_pairs = [], []

    for i in range(len(sig)):
        for j in range(i + 1, len(sig)):
            ci_lab = centroids[sig[i]]
            cj_lab = centroids[sig[j]]

            de = delta_e_ab(ci_lab, cj_lab)
            dl = delta_l_star(ci_lab, cj_lab)

            is_distinct = de >= min_delta_e
            is_low      = dl < max_delta_l
            is_prob     = is_distinct and is_low
            weight      = float(fracs[sig[i]] + fracs[sig[j]])

            pair = dict(
                color_a_rgb=cluster_info[i]["centroid_rgb"],
                color_b_rgb=cluster_info[j]["centroid_rgb"],
                delta_e=round(de, 2),
                delta_l_star=round(dl, 2),
                pair_weight=round(weight, 4),
                is_problematic=bool(is_prob),
            )
            all_pairs.append(pair)
            if is_prob:
                problematic_pairs.append(dict(
                    color_a_rgb=pair["color_a_rgb"],
                    color_b_rgb=pair["color_b_rgb"],
                    delta_e=pair["delta_e"],
                    delta_l_star=pair["delta_l_star"],
                ))

    # Verdict
    if problematic_pairs:
        descs = []
        for p in problematic_pairs:
            descs.append(
                f"RGB{tuple(p['color_a_rgb'])} vs "
                f"RGB{tuple(p['color_b_rgb'])} "
                f"(dE={p['delta_e']}, dL*={p['delta_l_star']})")
        reason = (f"{len(problematic_pairs)} color pair(s) are "
                  f"chromatically distinct but indistinguishable in "
                  f"grayscale: " + "; ".join(descs))
        verdict = "PROBLEMATIC"
    else:
        verdict = "ACCESSIBLE"
        reason  = (f"All {len(all_pairs)} color pair(s) have sufficient "
                   f"luminance contrast (dL* >= {max_delta_l}) in "
                   f"grayscale.")

    return dict(
        verdict=verdict,
        n_colors=len(sig),
        problematic_pairs=problematic_pairs,
        all_pairs=all_pairs,
        reason=reason,
    )


### JSON INPUT PARSING

def _parse_figures(json_path, analyze_all=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    figures = []

    # Classification JSON (has "classifications" with "class" field)
    if "classifications" in data:
        for c in data["classifications"]:
            fig = dict(figure_id=c.get("figure_id", "?"),
                       bbox=c["bbox"],
                       cls=c.get("class", "unknown"))
            if analyze_all or fig["cls"] == "discrete":
                figures.append(fig)

    # Detection JSON (has "detections", no class labels)
    elif "detections" in data:
        if not analyze_all:
            print("  WARNING: Detection JSON has no class labels. "
                  "Use --analyze-all to process all figures.")
        for d in data["detections"]:
            fig = dict(figure_id=d.get("figure_id", "?"),
                       bbox=d["bbox"],
                       cls="discrete")
            if analyze_all:
                figures.append(fig)

    return figures, data


def _full_image_figure(img_shape):
    h, w = img_shape[:2]
    return [dict(
        figure_id="full_image",
        bbox=dict(x1=0, y1=0, x2=w, y2=h),
        cls="unknown",
    )]


### IMAGE-LEVEL PROCESSING

def process_image(image_path, json_path, *, visualize=False,
                  analyze_all=False, min_delta_e=MIN_DELTA_E,
                  max_delta_l=MAX_DELTA_L):
    image_path = Path(image_path)
    json_path  = Path(json_path) if json_path is not None else None

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image {image_path}")
        return None

    h, w = img.shape[:2]

    if json_path is None:
        figures = _full_image_figure(img.shape)
        print("  No JSON provided: analysing full image as one figure\n")
    else:
        figures, raw_data = _parse_figures(json_path, analyze_all)
        if not figures:
            print(f"  No discrete figures to analyse in {json_path.name}\n")
            return None

    print(f"Analysing: {image_path.name}")
    print(f"  Discrete figures to analyse: {len(figures)}\n")

    # Greyscale canvas for visualisation
    if visualize:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis_img = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)

    analysis_results = []
    summary = dict(total_discrete=len(figures), accessible=0, problematic=0)

    for fig in figures:
        fid  = fig["figure_id"]
        bbox = fig["bbox"]

        x1, y1 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"]))
        x2, y2 = min(w, int(bbox["x2"])), min(h, int(bbox["y2"]))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"  Figure {fid}: empty crop, skipping\n")
            continue

        result = analyze_contrast_loss(
            crop, min_delta_e=min_delta_e,
            max_delta_l=max_delta_l)

        verdict = result["verdict"]

        # Console output
        print(f"  Figure {fid}: {verdict}")
        print(f"    Colors detected : {result['n_colors']}")
        print(f"    Reason           : {result['reason']}")
        if result["problematic_pairs"]:
            for pp in result["problematic_pairs"]:
                print(
                    f"      >> {pp['color_a_rgb']} vs {pp['color_b_rgb']}"
                    f"  dE={pp['delta_e']}  dL*={pp['delta_l_star']}")
        elif result.get("all_pairs"):
            print("      Pair diagnostics (non-problematic):")
            for pp in result["all_pairs"]:
                print(
                    f"      .. {pp['color_a_rgb']} vs {pp['color_b_rgb']}"
                    f"  dE={pp['delta_e']}  dL*={pp['delta_l_star']}")
        print()

        summary["problematic" if verdict == "PROBLEMATIC"
                else "accessible"] += 1

        analysis_results.append(dict(
            figure_id=fid,
            n_colors=result["n_colors"],
            problematic_pairs=result["problematic_pairs"],
            all_pairs=result.get("all_pairs", []),
        ))

        # Draw on visualisation
        if visualize:
            color = (COLOR_PROBLEMATIC if verdict == "PROBLEMATIC"
                     else COLOR_ACCESSIBLE)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2),
                          color, BBOX_THICKNESS)

            label = f"Fig {fid}: {verdict}"
            (tw, th), _ = cv2.getTextSize(label, FONT,
                                          FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(vis_img,
                          (x1, y1 - th - 12), (x1 + tw + 10, y1),
                          color, -1)
            cv2.putText(vis_img, label, (x1 + 5, y1 - 7),
                        FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

    # Summary
    print(f"  SUMMARY")
    print(f"    Total discrete : {summary['total_discrete']}")
    print(f"    Accessible     : {summary['accessible']}")
    print(f"    Problematic    : {summary['problematic']}")

    # Save JSON report
    output_json = dict(
        image_name=image_path.name,
        timestamp=datetime.now().isoformat(),
        summary=summary,
        analyses=analysis_results,
    )

    json_out = OUTPUT_DIR / f"{image_path.stem}_contrast_analysis.json"
    with open(json_out, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  JSON saved: {json_out}")

    # Save visualisation
    if visualize:
        vis_path = OUTPUT_DIR / f"{image_path.stem}_contrast_visualized.png"
        cv2.imwrite(str(vis_path), vis_img)
        print(f"  Visualisation saved: {vis_path}")

    print()
    return output_json


def _save_combined_folder_report(image_folder, reports, *, json_folder=None,
                                 min_delta_e=MIN_DELTA_E,
                                 max_delta_l=MAX_DELTA_L):
    if not reports:
        return None

    total_figures = sum(r["summary"]["total_discrete"] for r in reports)
    total_accessible = sum(r["summary"]["accessible"] for r in reports)
    total_problematic = sum(r["summary"]["problematic"] for r in reports)

    combined = dict(
        image_folder=str(image_folder),
        timestamp=datetime.now().isoformat(),
        summary=dict(
            total_images=len(reports),
            total_figures=total_figures,
            accessible=total_accessible,
            problematic=total_problematic,
        ),
        images=reports,
    )

    base_name = Path(image_folder).name
    if json_folder is not None:
        base_name = f"{base_name}__{Path(json_folder).name}"

    out_path = OUTPUT_DIR / f"{base_name}_contrast_analysis_combined.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n  FOLDER SUMMARY")
    print(f"    Images processed : {len(reports)}")
    print(f"    Figures total    : {total_figures}")
    print(f"    Accessible       : {total_accessible}")
    print(f"    Problematic      : {total_problematic}")
    print(f"  Combined JSON saved: {out_path}\n")

    return out_path


### FOLDER-LEVEL PROCESSING

def process_folder(image_folder, json_folder, **kwargs):
    image_folder = Path(image_folder)
    json_folder  = Path(json_folder)

    json_files = sorted(json_folder.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_folder}")
        return

    print(f"Found {len(json_files)} JSON file(s)\n")
    reports = []

    for jf in json_files:
        stem = jf.stem
        # strip common suffixes to recover the original image stem
        for sfx in ("_classifications", "_detections", "_classified",
                     "_contrast_analysis"):
            if stem.endswith(sfx):
                stem = stem[: -len(sfx)]
                break

        image_file = None
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidate = image_folder / f"{stem}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            print(f"WARNING: No image found for {jf.name}, skipping\n")
            continue

        report = process_image(image_file, jf, **kwargs)
        if report is not None:
            reports.append(report)

    _save_combined_folder_report(
        image_folder,
        reports,
        json_folder=json_folder,
        min_delta_e=kwargs.get("min_delta_e", MIN_DELTA_E),
        max_delta_l=kwargs.get("max_delta_l", MAX_DELTA_L),
    )


def process_folder_without_json(image_folder, **kwargs):
    image_folder = Path(image_folder)

    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    image_files = sorted(
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix in exts
    )

    if not image_files:
        print(f"No image files found in {image_folder}")
        return

    print(f"Found {len(image_files)} image file(s)\n")
    reports = []
    for image_file in image_files:
        report = process_image(image_file, None, **kwargs)
        if report is not None:
            reports.append(report)

    _save_combined_folder_report(
        image_folder,
        reports,
        json_folder=None,
        min_delta_e=kwargs.get("min_delta_e", MIN_DELTA_E),
        max_delta_l=kwargs.get("max_delta_l", MAX_DELTA_L),
    )


### ENTRY POINT

def main():
    parser = build_parser()
    args = parser.parse_args()

    print("  JetFighter - Contrast Loss Analysis (Step 3)")
    print(f"  Thresholds: dE_min={args.delta_e}, "
          f"dL*_max={args.delta_l}")
    print()

    img_path = Path(args.image)
    det_path = Path(args.detections) if args.detections else None

    if not img_path.exists():
        print(f"ERROR: Path not found: {img_path}")
        return 1
    if det_path is not None and not det_path.exists():
        print(f"ERROR: Path not found: {det_path}")
        return 1

    common = dict(visualize=args.visualize, analyze_all=args.analyze_all,
                  min_delta_e=args.delta_e, max_delta_l=args.delta_l)

    if img_path.is_file():
        if det_path is None:
            process_image(img_path, None, **common)
        else:
            if not det_path.is_file():
                print(f"ERROR: Expected a JSON file, got: {det_path}")
                return 1
            process_image(img_path, det_path, **common)

    elif img_path.is_dir():
        if det_path is None:
            process_folder_without_json(img_path, **common)
        else:
            if not det_path.is_dir():
                print(f"ERROR: Expected a JSON folder, got: {det_path}")
                return 1
            process_folder(img_path, det_path, **common)

    else:
        print(f"ERROR: Invalid path: {img_path}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
