import { useRef, useState, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import FigureOverlay from "./FigureOverlay";

export default function AnalysisViewer({ data, currentPage, onPageChange }) {
  const imgRef = useRef(null);
  const [dims, setDims] = useState({
    dw: 0,
    dh: 0,
    nw: 1,
    nh: 1,
  });

  const page = data.pages[currentPage];

  /* measure displayed image size */
  const measure = () => {
    const el = imgRef.current;
    if (!el) return;
    setDims({
      dw: el.clientWidth,
      dh: el.clientHeight,
      nw: el.naturalWidth || 1,
      nh: el.naturalHeight || 1,
    });
  };

  useEffect(() => {
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const scaleX = dims.dw / dims.nw;
  const scaleY = dims.dh / dims.nh;

  const hasPrev = currentPage > 0;
  const hasNext = currentPage < data.total_pages - 1;

  return (
    <div className="flex flex-col items-center w-full max-w-5xl gap-4">
      {/* ── Summary bar ───────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-4 text-sm px-4 py-2 rounded-lg bg-gray-100/70 dark:bg-gray-900/70 border border-gray-300 dark:border-gray-800 w-full justify-center">
        <span className="font-medium text-gray-700 dark:text-gray-300 truncate max-w-xs">
          {data.filename}
        </span>
        <span className="text-gray-400 dark:text-gray-500">|</span>
        <span className="text-gray-600 dark:text-gray-400">
          {data.total_figures} figure{data.total_figures !== 1 ? "s" : ""}
        </span>

        {data.problematic_figures > 0 && (
          <span className="flex items-center gap-1 text-red-500 dark:text-red-400">
            <AlertTriangle size={14} /> {data.problematic_figures} problematic
          </span>
        )}
        <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
          <CheckCircle size={14} /> {data.safe_figures} safe
        </span>
      </div>

      {/* ── Page viewer ─────────────────────────────── */}
      <div className="relative flex items-center gap-2 w-full justify-center">
        {/* Prev */}
        <button
          onClick={() => hasPrev && onPageChange(currentPage - 1)}
          disabled={!hasPrev}
          className="shrink-0 p-2 rounded-full bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition"
        >
          <ChevronLeft size={20} />
        </button>

        {/* Image + overlays */}
        <div className="relative inline-block">
          <img
            ref={imgRef}
            src={page.image_url}
            alt={`Page ${page.page_number}`}
            onLoad={measure}
            className="rounded-lg shadow-2xl"
            style={{ maxHeight: "calc(100vh - 220px)", maxWidth: "100%" }}
            draggable={false}
          />

          {/* figure overlays */}
          {page.figures.map((fig) => (
            <FigureOverlay
              key={fig.figure_id}
              figure={fig}
              scaleX={scaleX}
              scaleY={scaleY}
              pageImageUrl={page.image_url}
              naturalWidth={dims.nw}
              naturalHeight={dims.nh}
            />
          ))}
        </div>

        {/* Next */}
        <button
          onClick={() => hasNext && onPageChange(currentPage + 1)}
          disabled={!hasNext}
          className="shrink-0 p-2 rounded-full bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition"
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* ── Page counter ───────────────────────────── */}
      <div className="flex items-center gap-3 text-sm text-gray-600 dark:text-gray-500">
        <span>
          Page {page.page_number} of {data.total_pages}
        </span>
        {page.num_figures > 0 && (
          <>
            <span>·</span>
            <span>
              {page.num_figures} figure{page.num_figures !== 1 ? "s" : ""} on
              this page
            </span>
          </>
        )}
      </div>
    </div>
  );
}
