import { useState, useEffect } from "react";
import axios from "axios";
import UploadZone from "./components/UploadZone";
import AnalysisViewer from "./components/AnalysisViewer";
import { Loader2, AlertCircle, RotateCcw, Moon, Sun } from "lucide-react";

export default function App() {
  const [state, setState] = useState("idle"); // idle | uploading | analyzing | done | error
  const [progress, setProgress] = useState(0);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [currentPage, setCurrentPage] = useState(0);
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleFile = async (file) => {
    setState("uploading");
    setProgress(0);
    setError("");

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await axios.post("/api/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const pct = Math.round((e.loaded * 100) / (e.total || 1));
          setProgress(pct);
          if (pct >= 100) setState("analyzing");
        },
      });
      setData(res.data);
      setCurrentPage(0);
      setState("done");
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Analysis failed."
      );
      setState("error");
    }
  };

  const reset = () => {
    setState("idle");
    setData(null);
    setError("");
    setProgress(0);
  };

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-800 bg-gray-50/70 dark:bg-gray-900/70 backdrop-blur-sm">
        <h1
          className="text-xl font-bold tracking-tight cursor-pointer"
          onClick={reset}
        >
          <span
            className="bg-gradient-to-r from-red-500 via-yellow-400 via-green-400 to-cyan-500 bg-clip-text text-transparent font-extrabold"
            style={{ WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}
          >
            Jet
          </span>
          <span className="text-indigo-600 dark:text-indigo-400">Fighter</span>
        </h1>

        {/* Center - Upload new button */}
        <div className="absolute left-1/2 -translate-x-1/2">
          {state === "done" && (
            <button
              onClick={reset}
              className="flex items-center gap-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition"
            >
              <RotateCcw size={14} /> Upload new
            </button>
          )}
        </div>

        {/* Right - Theme toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800 transition"
          aria-label="Toggle theme"
        >
          {darkMode ? <Sun size={18} className="text-gray-400" /> : <Moon size={18} className="text-gray-600" />}
        </button>
      </header>

      {/* Body */}
      <main className="flex-1 flex items-center justify-center p-6">
        {state === "idle" && <UploadZone onFile={handleFile} />}

        {(state === "uploading" || state === "analyzing") && (
          <div className="flex flex-col items-center gap-4 text-gray-600 dark:text-gray-400">
            <Loader2 className="animate-spin text-indigo-400" size={48} />
            {state === "uploading" ? (
              <>
                <p className="text-lg">Uploading… {progress}%</p>
                <div className="w-64 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 transition-all duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </>
            ) : (
              <p className="text-lg">
                Analyzing figures… this may take a moment
              </p>
            )}
          </div>
        )}

        {state === "error" && (
          <div className="flex flex-col items-center gap-4 max-w-md text-center">
            <AlertCircle className="text-red-400" size={48} />
            <p className="text-red-300">{error}</p>
            <button
              onClick={reset}
              className="mt-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition"
            >
              Try again
            </button>
          </div>
        )}

        {state === "done" && data && (
          <AnalysisViewer
            data={data}
            currentPage={currentPage}
            onPageChange={setCurrentPage}
          />
        )}
      </main>
    </div>
  );
}
