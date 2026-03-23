import { useCallback, useState } from "react";
import { Upload, FileText } from "lucide-react";

export default function UploadZone({ onFile }) {
  const [dragOver, setDragOver] = useState(false);

  const ALLOWED_TYPES = ["application/pdf", "image/png", "image/jpeg"];

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file && ALLOWED_TYPES.includes(file.type)) onFile(file);
    },
    [onFile]
  );

  const handleInput = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`
        flex flex-col items-center justify-center gap-4
        w-full max-w-lg aspect-[4/3] rounded-2xl border-2 border-dashed
        cursor-pointer transition-all duration-200
        ${
          dragOver
            ? "border-indigo-400 bg-indigo-500/10 scale-[1.02]"
            : "border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500 bg-gray-100/40 dark:bg-gray-900/40"
        }
      `}
    >
      <input
        type="file"
        accept=".pdf,.png,.jpg,.jpeg"
        onChange={handleInput}
        className="hidden"
      />

      <div className="w-16 h-16 rounded-full bg-gray-200 dark:bg-gray-800 flex items-center justify-center">
        {dragOver ? (
          <FileText className="text-indigo-600 dark:text-indigo-400" size={28} />
        ) : (
          <Upload className="text-gray-600 dark:text-gray-400" size={28} />
        )}
      </div>

      <div className="text-center px-4">
        <p className="text-gray-700 dark:text-gray-300 font-medium">
          Drop a PDF or image here or <span className="text-indigo-600 dark:text-indigo-400">browse</span>
        </p>
        <p className="text-gray-500 dark:text-gray-500 text-sm mt-1">
          Automated detection of Accessible and Problematic Visualizations
        </p>
      </div>
    </label>
  );
}
