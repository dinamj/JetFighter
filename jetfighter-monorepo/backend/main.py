"""
JetFighter — FastAPI Backend
POST /api/analyze   Upload a PDF or image -> run pipeline -> return JSON.
GET  /api/static/*  Serve page images.
GET  /api/health    Readiness check.
"""

import os
import io
import uuid
import shutil
import traceback
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline import JetFighterPipeline

### Paths
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent          # …/JetFighter
MODELS_DIR   = PROJECT_ROOT / "models"
STATIC_DIR   = BACKEND_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "24"))


def _cleanup_expired_sessions(static_dir: Path, ttl_hours: int) -> int:
    # Delete session folders older than the configured TTL
    if ttl_hours <= 0:
        return 0

    now_ts = datetime.now().timestamp()
    ttl_seconds = ttl_hours * 3600
    removed = 0

    for entry in static_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            age_seconds = now_ts - entry.stat().st_mtime
            if age_seconds > ttl_seconds:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except Exception:
            continue

    return removed


def _resolve_frontend_dir() -> Path | None:
    candidates = [
        BACKEND_DIR / "static_frontend",
        BACKEND_DIR / "static_frontend" / "dist",
    ]
    for d in candidates:
        if (d / "index.html").exists():
            return d
    return None


FRONTEND_DIR = _resolve_frontend_dir()

# Poppler path for pdf2image on Windows (set env var if needed)
POPPLER_PATH = os.environ.get("POPPLER_PATH", None)

# Auto-detect Poppler if not set (Windows)
if POPPLER_PATH is None and os.name == 'nt':
    possible_paths = [
        PROJECT_ROOT / "Release-25.12.0-0" / "poppler-25.12.0" / "Library" / "bin",
        Path("C:/poppler/Library/bin"),
    ]
    for p in possible_paths:
        if p.exists():
            POPPLER_PATH = str(p)
            break

### App
app = FastAPI(title="JetFighter API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

### Pipeline singleton
pipeline: JetFighterPipeline | None = None


@app.on_event("startup")
def startup():
    global pipeline
    print(f"[JetFighter] Project root : {PROJECT_ROOT}")
    print(f"[JetFighter] Models dir   : {MODELS_DIR}")
    cleaned = _cleanup_expired_sessions(STATIC_DIR, SESSION_TTL_HOURS)
    print(f"[JetFighter] Static cleanup: removed {cleaned} expired session folder(s) (TTL={SESSION_TTL_HOURS}h)")
    pipeline = JetFighterPipeline(models_dir=MODELS_DIR)
    print("[JetFighter] Pipeline ready ✓")
    if FRONTEND_DIR is not None:
        print("[JetFighter] UI: http://localhost:8000")
    else:
        print("[JetFighter] UI not bundled (root / returns info JSON)")


### Endpoints

@app.get("/api/health")
def health():
    return dict(
        status="ok",
        timestamp=datetime.now().isoformat(),
        models=dict(
            detector=pipeline.detector is not None if pipeline else False,
            classifier=pipeline.classifier is not None if pipeline else False,
        ),
    )


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@app.post("/api/analyze")
def analyze_file(file: UploadFile = File(...)):
    _cleanup_expired_sessions(STATIC_DIR, SESSION_TTL_HOURS)

    filename = file.filename.lower()
    ext = Path(filename).suffix
    is_pdf = ext == ".pdf"
    is_image = ext in ALLOWED_IMAGE_EXTS

    if not is_pdf and not is_image:
        raise HTTPException(400, detail="Only PDF, PNG, JPG and JPEG files are accepted.")

    session_id = uuid.uuid4().hex[:8]
    session_dir = STATIC_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_bytes = file.file.read()

        if is_pdf:
            # Convert PDF to page images
            try:
                from pdf2image import convert_from_bytes
                kwargs = dict(dpi=200)
                if POPPLER_PATH:
                    kwargs["poppler_path"] = POPPLER_PATH
                pil_images = convert_from_bytes(file_bytes, **kwargs)
            except ImportError:
                raise HTTPException(500, detail="pdf2image is not installed.")
            except Exception as exc:
                raise HTTPException(500, detail=f"PDF conversion failed: {exc}")

            # Analyse each page (detect + classify + contrast)
            pages = []
            for idx, pil_img in enumerate(pil_images, 1):
                fname = f"page_{idx}.png"
                page_path = session_dir / fname
                pil_img.save(str(page_path), "PNG")

                result = pipeline.analyze_page(page_path)
                pages.append(dict(
                    page_number=idx,
                    image_url=f"/api/static/{session_id}/{fname}",
                    image_width=result.get("image_width", pil_img.width),
                    image_height=result.get("image_height", pil_img.height),
                    figures=result.get("figures", []),
                    num_figures=result.get("num_figures", 0),
                ))

            # Renumber figures globally
            global_figure_id = 1
            for page in pages:
                for fig in page["figures"]:
                    fig["figure_id"] = global_figure_id
                    global_figure_id += 1

        else:
            # Single image: classify only (no detection)
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
            fname = f"image_1.png"
            img_path = session_dir / fname
            pil_img.save(str(img_path), "PNG")

            result = pipeline.analyze_image(img_path)
            pages = [dict(
                page_number=1,
                image_url=f"/api/static/{session_id}/{fname}",
                image_width=result.get("image_width", pil_img.width),
                image_height=result.get("image_height", pil_img.height),
                figures=result.get("figures", []),
                num_figures=result.get("num_figures", 0),
            )]

        # Summary stats
        total  = sum(p["num_figures"] for p in pages)
        n_red  = sum(1 for p in pages for f in p["figures"] if f["status"] == "red")
        n_green = sum(1 for p in pages for f in p["figures"] if f["status"] == "green")

        return JSONResponse(dict(
            session_id=session_id,
            filename=file.filename,
            total_pages=len(pages),
            total_figures=total,
            problematic_figures=n_red,
            safe_figures=n_green,
            timestamp=datetime.now().isoformat(),
            pages=pages,
        ))

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(500, detail="Analysis failed – see server logs.")


### Serve frontend dist in production (Docker)
if FRONTEND_DIR is not None:
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def root_info():
        return JSONResponse(
            status_code=404,
            content=dict(
                detail="Frontend not found in container.",
                expected=[
                    str(BACKEND_DIR / "static_frontend" / "index.html"),
                    str(BACKEND_DIR / "static_frontend" / "dist" / "index.html"),
                ],
                health="/api/health",
            ),
        )
