# JetFighter Monorepo

This is a full-stack application for analysing scientific paper PDFs and images for color-accessibility issues.

## 1. Prerequisites (First Time Setup)

## Install Miniforge
Download and install **Miniforge** from here: [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge). This allows you to manage the Python environment.

## Set up the Environment
Open a terminal (PowerShell) and navigate to the main **JetFighter** folder (where `environment.yml` is).

1. **Create the environment**:
   - If you have an NVIDIA GPU (CUDA):
     ```powershell
     conda env create -f environment_cuda.yml
     ```
   - If you are on CPU or macOS:
     ```powershell
     conda env create -f environment.yml
     ```

2. **Activate the environment**:
   ```powershell
   conda activate jetfighter
   ```

## Set up Poppler
Poppler is required to convert PDF pages into images.
It is already downloaded in this folder (`Release-25.12.0-0`). You just need to tell your computer where to find it.

### Temporary (per-session) option

Run this command in your PowerShell (every time you open a new terminal for the backend):
```powershell
$env:PATH += ";$PWD\Release-25.12.0-0\poppler-25.12.0\Library\bin"
```

### Permanent option

This sets the User PATH. Replace `<path-to-project>` with the path to your JetFighter folder (or the folder where you placed the `Release-...` directory):

```powershell
[Environment]::SetEnvironmentVariable("PATH", [Environment]::GetEnvironmentVariable("PATH", "User") + ";<path-to-project>\Release-25.12.0-0\poppler-25.12.0\Library\bin", "User")
```

After running this command, close and reopen your terminal to ensure the new PATH is active.

---

## 2. Running the Application

You need **two separate terminals** running at the same time: one for the Backend (Python/AI) and one for the Frontend (Website).

### Terminal 1: Backend
The backend runs the AI models.

1. Open a new terminal.
2. Activate the environment:
   ```powershell
   conda activate jetfighter
   ```
3. Add Poppler to path (if not permanently set):
   ```powershell
   $env:PATH += ";<path-to-project>\Release-25.12.0-0\poppler-25.12.0\Library\bin"
   ```
4. Navigate to the backend folder:
   ```powershell
   cd jetfighter-monorepo/backend
   ```
5. Install Python dependencies (only needed once):
   ```powershell
   pip install -r requirements.txt
   ```
6. Start the server:
   ```powershell
   uvicorn main:app --reload --port 8000
   ```

### Terminal 2: Frontend
The frontend is the user interface you see in the browser.

1. Open a second terminal
2. Navigate to the frontend folder:
   ```powershell
   cd jetfighter-monorepo/frontend
   ```
3. Install Node.js dependencies (only needed once):
   ```powershell
   npm install
   ```
4. Start the website:
   ```powershell
   npm run dev
   ```

Open **http://localhost:5173** in your browser.

---

## 3. Optional: Run with Docker

You can run JetFighter with Docker if you prefer not to set up Python/Node manually.

### Requirements

1. Install Docker Desktop (required):
   - https://www.docker.com/products/docker-desktop/
2. Make sure Docker Desktop is running.

### Important path note

The current `Dockerfile` and `docker-compose.yml` are located in the main JetFighter root folder (one level above `jetfighter-monorepo`).

If you are currently inside `jetfighter-monorepo`, first go to the project root:

```powershell
cd ..
```

### Build and start

From the JetFighter root folder:

```powershell
docker compose build
docker compose up -d
```

Open:

- http://localhost:8000

### Stop containers

```powershell
docker compose down
```

### Without Docker

If you do not want Docker, use the Conda setup above:

- `environment_cuda.yml` for NVIDIA GPU/CUDA
- `environment.yml` for CPU or macOS

