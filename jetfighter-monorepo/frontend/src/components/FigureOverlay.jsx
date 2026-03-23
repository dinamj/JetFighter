import { useState, useRef, useEffect } from "react";

const CATEGORY_LABELS = {
  rainbow_gradient: "Rainbow Gradient",
  safe_gradient: "Safe Gradient",
  accessible_discrete: "Accessible",
  problematic_discrete: "Poor Contrast",
};

// red/green palette with labels (good contrast)
const COLORS = {
  problem: { border: "rgb(239,68,68)",  label: "⚠ " },
  safe:    { border: "rgb(16,185,129)", label: "✓ " },
};

export default function FigureOverlay({ figure, scaleX, scaleY, pageImageUrl, naturalWidth, naturalHeight }) {
  const [hover, setHover] = useState(false);
  const [grayCrop, setGrayCrop] = useState(null);
  const canvasRef = useRef(null);

  const { bbox, status, reason, category, contrast_details, figure_id, classification } =
    figure;

  const isProblem = status === "red";
  const palette = isProblem ? COLORS.problem : COLORS.safe;
  const borderColor = palette.border;
  const bgColor = palette.bg;

  const left = bbox.x1 * scaleX;
  const top = bbox.y1 * scaleY;
  const width = (bbox.x2 - bbox.x1) * scaleX;
  const height = (bbox.y2 - bbox.y1) * scaleY;

  // Show grayscale preview for discrete figures
  const isDiscrete = classification === "discrete";

  // Generate grayscale crop on hover
  useEffect(() => {
    if (!hover || !isDiscrete || !pageImageUrl) {
      setGrayCrop(null);
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const cropW = bbox.x2 - bbox.x1;
      const cropH = bbox.y2 - bbox.y1;
      canvas.width = cropW;
      canvas.height = cropH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, bbox.x1, bbox.y1, cropW, cropH, 0, 0, cropW, cropH);

      // Convert BGR→LAB L* channel (same as backend pipeline)
      const imageData = ctx.getImageData(0, 0, cropW, cropH);
      const d = imageData.data;
      
      // RGB → LAB conversion (simplified, matches OpenCV cv2.COLOR_BGR2LAB L-channel)
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i] / 255.0;
        const g = d[i + 1] / 255.0;
        const b = d[i + 2] / 255.0;
        
        // RGB → XYZ (D65 illuminant)
        const rLin = r <= 0.04045 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
        const gLin = g <= 0.04045 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
        const bLin = b <= 0.04045 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
        
        const x = rLin * 0.4124564 + gLin * 0.3575761 + bLin * 0.1804375;
        const y = rLin * 0.2126729 + gLin * 0.7151522 + bLin * 0.0721750;
        const z = rLin * 0.0193339 + gLin * 0.1191920 + bLin * 0.9503041;
        
        // XYZ → LAB L* (matches CIELAB)
        const fy = y > 0.008856 ? Math.pow(y, 1/3) : (7.787 * y + 16/116);
        const L = 116 * fy - 16;
        
        // Map L* [0,100] to grayscale [0,255]
        const gray = Math.round(L * 2.55);
        d[i] = d[i + 1] = d[i + 2] = gray;
      }
      
      ctx.putImageData(imageData, 0, 0);
      setGrayCrop(canvas.toDataURL());
    };
    img.src = pageImageUrl;
  }, [hover, isDiscrete, pageImageUrl, bbox.x1, bbox.y1, bbox.x2, bbox.y2]);

  // Decide whether tooltip should open upward
  const flipTooltip = top + height + 260 > (window.innerHeight ?? 900);

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        border: `3px solid ${borderColor}`,
        backgroundColor: bgColor,
        zIndex: hover ? 40 : 10,
        pointerEvents: "auto",
      }}
      className="cursor-pointer transition-shadow"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {/* Label pill */}
      <div
        className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-semibold text-white rounded-t whitespace-nowrap select-none"
        style={{ backgroundColor: borderColor }}
      >
        {palette.label}Fig {figure_id}
      </div>

      {/* Tooltip */}
      {hover && (
        <div
          className="absolute z-50 flex gap-2"
          style={{
            top: 0,
            left: width + 6,
          }}
        >
          {/* Info panel */}
          <div
            className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white p-3 rounded-lg shadow-xl text-xs leading-relaxed"
            style={{ minWidth: 200, maxWidth: 300 }}
          >
            {/* Category */}
            <div className="flex items-center gap-2 mb-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: borderColor }}
              />
              <span className="font-bold text-sm">
                {CATEGORY_LABELS[category] ?? category}
              </span>
            </div>

            {/* Confidence */}
            <p className="text-gray-500 dark:text-gray-500 mb-2">
              Classification confidence:{" "}
              <span className="text-gray-700 dark:text-gray-300">
                {(figure.classification_confidence * 100).toFixed(1)}%
              </span>
            </p>

            {/* Contrast pairs (for discrete figures) */}
            {contrast_details?.problematic_pairs?.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-300 dark:border-gray-800">
                <p className="text-gray-600 dark:text-gray-500 mb-1.5">Conflicting color pairs:</p>
                <div className="flex flex-col gap-1.5">
                  {contrast_details.problematic_pairs.map((pair, i) => {
                    return (
                      <div key={i} className="flex items-center gap-2">
                        <span
                          className="w-4 h-4 rounded border border-white/20"
                          style={{
                            backgroundColor: `rgb(${pair.color_a_rgb.join(",")})`,
                          }}
                        />
                        <span className="text-gray-500 dark:text-gray-600">vs</span>
                        <span
                          className="w-4 h-4 rounded border border-white/20"
                          style={{
                            backgroundColor: `rgb(${pair.color_b_rgb.join(",")})`,
                          }}
                        />
                        <span className="text-gray-600 dark:text-gray-500 ml-1 text-[11px]">
                          ΔE = {pair.delta_e}, ΔL* = {pair.delta_l_star}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <p className="text-gray-500 dark:text-gray-600 text-[10px] mt-2 italic">
                  High ΔE (&gt; 35) + Low ΔL* (&lt; 10) → Problematic
                </p>
              </div>
            )}

            {/* Color swatches (for all discrete figures) */}
            {contrast_details?.cluster_info?.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-300 dark:border-gray-800">
                <p className="text-gray-600 dark:text-gray-500 mb-1.5">Detected colors:</p>
                <div className="flex flex-wrap gap-1.5">
                  {contrast_details.cluster_info.map((c, i) => (
                    <div key={i} className="flex items-center gap-1">
                      <span
                        className="w-4 h-4 rounded border border-white/20"
                        style={{
                          backgroundColor: `rgb(${c.centroid_rgb.join(",")})`,
                        }}
                      />
                      <span className="text-gray-500 dark:text-gray-600 text-[10px]">
                        {(c.fraction * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Grayscale preview for discrete figures */}
          {isDiscrete && grayCrop && (
            <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg shadow-xl overflow-hidden flex flex-col">
              <div className="px-2 py-1 text-[10px] font-semibold text-gray-500 dark:text-gray-400 text-center border-b border-gray-300 dark:border-gray-700">
                Grayscale Simulation
              </div>
              <img
                src={grayCrop}
                alt="Grayscale preview"
                className="block"
                style={{ maxWidth: 300, maxHeight: 300, objectFit: "contain" }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
