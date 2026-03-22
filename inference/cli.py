"""
JetFighter — CLI
Runs the full pipeline (detect -> classify -> contrast) on images or PDFs

Usage:
    python inference/cli.py image.png
    python inference/cli.py paper.pdf
    python inference/cli.py ./folder_of_pdfs/
    python inference/cli.py paper.pdf --visualize
    python inference/cli.py paper.pdf -o results/
"""

import sys
import os
import csv
import cv2
import tempfile
import shutil
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

# Add project root to path so we can improt the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "jetfighter-monorepo" / "backend"))

from pipeline import JetFighterPipeline, CATEGORY_INFO


### CONFIGURATION

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cli"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
PDF_EXTENSION = ".pdf"

# Visualization colors (BGR)
COLOR_RED   = (0, 0, 239)
COLOR_GREEN = (0, 185, 16)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CATEGORY_LABELS = {
    "rainbow_gradient":     "Rainbow Gradient",
    "safe_gradient":        "Safe Gradient",
    "accessible_discrete":  "Accessible",
    "problematic_discrete": "Poor Contrast",
}


### CLI

def build_parser():
    p = ArgumentParser(description="JetFighter CLI — analyse scientific figures for color accessibility")
    p.add_argument("input", help="Image file, PDF file, or folder")
    p.add_argument("-o", "--output", default=None, help="Output directory (default: outputs/cli/)")
    p.add_argument("--visualize", "-v", action="store_true", help="Generate annotated visualizations")
    p.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion (default: 200)")
    return p


### PDF CONVERSION

def pdf_to_images(pdf_path, output_dir, dpi=200):
    """Convert PDF pages to PNG images. Returns list of (page_number, image_path)."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image not installed. Run: pip install pdf2image")
        sys.exit(1)

    # find poppler on Windows
    poppler_path = os.environ.get("POPPLER_PATH", None)
    if poppler_path is None and os.name == "nt":
        candidates = [
            PROJECT_ROOT / "Release-25.12.0-0" / "poppler-25.12.0" / "Library" / "bin",
            Path("C:/poppler/Library/bin"),
        ]
        for p in candidates:
            if p.exists():
                poppler_path = str(p)
                break

    kwargs = dict(dpi=dpi)
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    pil_images = convert_from_path(str(pdf_path), **kwargs)
    pages = []
    for idx, pil_img in enumerate(pil_images, 1):
        fname = f"{pdf_path.stem}_page_{idx}.png"
        page_path = output_dir / fname
        pil_img.save(str(page_path), "PNG")
        pages.append((idx, page_path))

    return pages


### VISUALIZATION

def draw_visualization(img, figures, output_path):
    """Draw bounding boxes + info panel on the image, similar to frontend overlay."""
    vis = img.copy()
    h_img, w_img = vis.shape[:2]

    for fig in figures:
        bbox = fig["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        is_problem = fig["status"] == "red"
        color = COLOR_RED if is_problem else COLOR_GREEN

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        # Label pill above bbox
        label = f"Fig {fig['figure_id']}: {CATEGORY_LABELS.get(fig['category'], fig['category'])}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), FONT, 0.5, (255, 255, 255), 1)

        # Info panel to the right of bbox
        panel_x = min(x2 + 8, w_img - 300)
        panel_y = max(y1, 10)
        lines = [
            f"Category: {CATEGORY_LABELS.get(fig['category'], fig['category'])}",
            f"Status: {'PROBLEMATIC' if is_problem else 'OK'}",
            f"Confidence: {fig['classification_confidence'] * 100:.1f}%",
        ]

        # Contrast details for discrete figures
        cd = fig.get("contrast_details")
        if cd and cd.get("problematic_pairs"):
            lines.append(f"Conflicting pairs: {len(cd['problematic_pairs'])}")
            for pair in cd["problematic_pairs"][:3]:
                lines.append(
                    f"  RGB{tuple(pair['color_a_rgb'])} vs RGB{tuple(pair['color_b_rgb'])}"
                    f"  dE={pair['delta_e']} dL*={pair['delta_l_star']}"
                )

        if cd and cd.get("cluster_info"):
            n_colors = len(cd["cluster_info"])
            lines.append(f"Detected colors: {n_colors}")

        # Draw panel background
        line_h = 18
        panel_h = len(lines) * line_h + 12
        panel_w = 290
        overlay = vis.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
        cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), color, 1)

        for i, line in enumerate(lines):
            ty = panel_y + 14 + i * line_h
            cv2.putText(vis, line, (panel_x + 6, ty), FONT, 0.4, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), vis)


### CSV REPORT

def write_csv_report(all_results, output_path):
    fieldnames = [
        "file", "page", "figure_id",
        "category", "status", "reason",
        "classification", "classification_confidence",
        "detection_confidence",
        "n_problematic_pairs", "n_colors",
    ]

    rows = []
    for entry in all_results:
        for fig in entry["figures"]:
            cd = fig.get("contrast_details") or {}
            rows.append({
                "file": entry["source_file"],
                "page": entry.get("page_number", ""),
                "figure_id": fig["figure_id"],
                "category": fig["category"],
                "status": "PROBLEMATIC" if fig["status"] == "red" else "OK",
                "reason": fig["reason"],
                "classification": fig["classification"],
                "classification_confidence": round(fig["classification_confidence"], 4),
                "detection_confidence": fig.get("detection_confidence", ""),
                "n_problematic_pairs": len(cd.get("problematic_pairs", [])),
                "n_colors": cd.get("n_colors", ""),
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


### MAIN

def main():
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        sys.exit(1)

    # Init pipeline
    print(f"Loading models from {MODELS_DIR} ...")
    pipeline = JetFighterPipeline(models_dir=MODELS_DIR)
    print("Pipeline ready.\n")

    # Collect input files
    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        for f in sorted(input_path.iterdir()):
            ext = f.suffix.lower()
            if ext in IMAGE_EXTENSIONS or ext == PDF_EXTENSION:
                files.append(f)
    else:
        print(f"ERROR: Invalid path: {input_path}")
        sys.exit(1)

    if not files:
        print(f"No supported files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to analyse.\n")

    # Track all results for CSV
    all_results = []
    global_fig_id = 1
    total_red = 0
    total_green = 0

    for file_path in files:
        ext = file_path.suffix.lower()
        print(f"--- {file_path.name} ---")

        if ext == PDF_EXTENSION:
            # PDF: convert pages to temp dir, analyse, then clean up
            tmp_dir = tempfile.mkdtemp(prefix="jetfighter_pages_")
            try:
                pages = pdf_to_images(file_path, Path(tmp_dir), dpi=args.dpi)
                print(f"  Converted {len(pages)} page(s)")

                for page_num, page_path in pages:
                    result = pipeline.analyze_page(page_path)
                    n_figs = result["num_figures"]

                    # Renumber globally
                    for fig in result["figures"]:
                        fig["figure_id"] = global_fig_id
                        global_fig_id += 1

                    n_red = sum(1 for f in result["figures"] if f["status"] == "red")
                    n_green = n_figs - n_red
                    total_red += n_red
                    total_green += n_green

                    if n_figs > 0:
                        print(f"  Page {page_num}: {n_figs} figure(s)  "
                              f"({n_red} problematic, {n_green} ok)")
                    else:
                        print(f"  Page {page_num}: no figures detected")

                    if args.visualize and n_figs > 0:
                        img = cv2.imread(str(page_path))
                        vis_path = output_dir / f"{file_path.stem}_p{page_num}_vis.png"
                        draw_visualization(img, result["figures"], vis_path)
                        print(f"    -> {vis_path.name}")

                    all_results.append({
                        "source_file": file_path.name,
                        "page_number": page_num,
                        "figures": result["figures"],
                    })
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        elif ext in IMAGE_EXTENSIONS:
            # Single image: no detection, classify whole image
            result = pipeline.analyze_image(str(file_path))

            for fig in result["figures"]:
                fig["figure_id"] = global_fig_id
                global_fig_id += 1

            n_figs = result["num_figures"]
            n_red = sum(1 for f in result["figures"] if f["status"] == "red")
            n_green = n_figs - n_red
            total_red += n_red
            total_green += n_green

            for fig in result["figures"]:
                status_str = "PROBLEMATIC" if fig["status"] == "red" else "OK"
                print(f"  {CATEGORY_LABELS.get(fig['category'], fig['category'])} "
                      f"({status_str}, conf={fig['classification_confidence']:.1%})")

            if args.visualize:
                img = cv2.imread(str(file_path))
                vis_path = output_dir / f"{file_path.stem}_vis.png"
                draw_visualization(img, result["figures"], vis_path)
                print(f"  -> {vis_path.name}")

            all_results.append({
                "source_file": file_path.name,
                "figures": result["figures"],
            })

    # Write CSV report (dont overwrite existing files)
    csv_path = output_dir / "report.csv"
    if csv_path.exists():
        i = 1
        while (output_dir / f"report_{i}.csv").exists():
            i += 1
        csv_path = output_dir / f"report_{i}.csv"
    write_csv_report(all_results, csv_path)

    # Summary
    total_figs = global_fig_id - 1
    print(f"Total figures: {total_figs}")
    print(f"  Problematic: {total_red}")
    print(f"  OK:          {total_green}")
    print(f"Report: {csv_path}")
    if args.visualize:
        print(f"Visualizations saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
