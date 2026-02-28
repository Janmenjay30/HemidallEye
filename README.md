<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ONNX_Runtime-GPU-green?logo=onnx" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/GPU-GTX_1650_(4GB)-red?logo=nvidia" alt="GPU">
</p>

# Heimdall — Family vs. Stranger CCTV Recognition System

> *"The all-seeing guardian of your home."*

A **real-time, edge-deployed** face recognition system that distinguishes family members from strangers using CCTV/webcam feeds. Designed to run entirely on a **GTX 1650 (4 GB VRAM)** with zero cloud dependency — all processing and data stay local.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [VRAM Budget](#vram-budget-gtx-1650--4096-mb)
- [Mathematical Framework](#mathematical-framework)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Download](#model-download)
- [Usage](#usage)
  - [Enroll Family Members](#1-enroll-family-members)
  - [Run the System](#2-run-the-system)
  - [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Database Strategy](#database-strategy)
- [Project Structure](#project-structure)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Real-Time Processing** — 25–30 FPS on GTX 1650 with full pipeline
- **6-Stage Pipeline** — Person detection → Face detection → Alignment → Recognition → Decision → Alert
- **Temporal Consistency** — Requires 5 consecutive stranger frames before alerting (eliminates false positives)
- **Anti-Spoof Protection** — Laplacian variance + aspect ratio checks to detect photos/screens
- **Adaptive Frame Skipping** — Dynamically adjusts based on GPU load to maintain target FPS
- **VRAM-Aware** — Budgets GPU memory per model, keeps peak usage under 2.6 GB
- **Multiple Input Sources** — Webcam, RTSP streams, video files
- **Local-Only** — Zero cloud dependency; all data stays on your machine
- **Easy Enrollment** — Add family members via photos or live webcam capture
- **Visual Overlay** — Color-coded bounding boxes (green = family, red = stranger, orange = uncertain)

---

## Demo

```
┌──────────────────────────────────────────────┐
│  Heimdall Live View                          │
│                                              │
│  ┌────────┐     ┌────────┐                   │
│  │ 🟩     │     │ 🟥     │                   │
│  │ Alice  │     │STRANGER│                   │
│  │ 0.72   │     │ 0.21   │                   │
│  └────────┘     │streak:5│                   │
│                 └────────┘                   │
│                                              │
│  FPS: 27.4 | Persons: 2 | Faces: 2          │
└──────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RTSP / Webcam / Video File                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ OpenCV decode (DirectShow on Windows)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Frame Acquisition & Preprocessing                       │
│  • Adaptive frame skip based on GPU load                           │
│  • Letterbox resize to 640×640                                     │
│  • BGR → RGB, normalize to [0, 1], float32 tensor                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Person Detection (YOLOv11n — ONNX Runtime GPU)          │
│  • Detects COCO "person" class only → bounding box crops           │
│  • Confidence ≥ 0.45, NMS IoU = 0.5, min area 3000 px²            │
│  ~1.2 GB VRAM  |  ~8 ms/frame                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Face Detection & Alignment (SCRFD-10GF — ONNX)          │
│  • Runs ONLY inside person crops (not full frame)                  │
│  • 5-point landmark detection → affine-aligned 112×112 face        │
│  • Filters faces < 40×40 px                                       │
│  ~0.3 GB VRAM  |  ~3 ms/crop                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Face Recognition (ArcFace w600k_r50 — ONNX)             │
│  • Produces 512-D L2-normalized embedding vector                   │
│  • Compared against enrolled gallery via cosine similarity         │
│  ~0.5 GB VRAM  |  ~5 ms/face                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Decision Engine + Anti-Spoof                            │
│  • Three-tier classification: FAMILY / UNCERTAIN / STRANGER        │
│  • Temporal consistency: 5 consecutive stranger frames → alert     │
│  • Anti-spoof: Laplacian variance, aspect ratio checks             │
│  • IoU-based multi-object tracker for identity persistence         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — Alert & Logging                                         │
│  • Stranger snapshot saved with timestamp                          │
│  • Console warning with track details                              │
│  • 30-second cooldown per track to prevent alert spam              │
│  • All data stored locally — zero cloud dependency                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

| Stage | Component | Model | Input | Output |
|-------|-----------|-------|-------|--------|
| 1 | Frame Acquisition | — | Raw stream | Preprocessed 640×640 tensor |
| 2 | Person Detection | YOLOv11n (11 MB) | Full frame | Person bounding boxes |
| 3 | Face Detection | SCRFD-10GF (17 MB) | Person crops | 112×112 aligned faces + landmarks |
| 4 | Face Recognition | ArcFace w600k_r50 (174 MB) | Aligned faces | 512-D embeddings |
| 5 | Decision Engine | — | Embeddings + DB | FAMILY / UNCERTAIN / STRANGER |
| 6 | Alert System | — | Decisions | Logs, snapshots, notifications |

---

## VRAM Budget (GTX 1650 — 4096 MB)

| Component | VRAM (MB) | Notes |
|-----------|-----------|-------|
| YOLOv11n (person detection) | ~1,200 | Nano variant, 640×640 input |
| SCRFD-10GF (face detection) | ~300 | From InsightFace buffalo_l |
| ArcFace-R50 (recognition) | ~500 | ResNet-50 backbone |
| ONNX Runtime overhead | ~400 | CUDA context + workspace |
| Frame buffers | ~200 | I/O tensors, pre/post processing |
| **Total** | **~2,600** | **1.4 GB headroom** |

---

## Mathematical Framework

### Cosine Similarity

Given a probe embedding **p** ∈ ℝ⁵¹² and gallery embedding **g**ᵢ:

$$\text{sim}(\mathbf{p}, \mathbf{g}_i) = \frac{\mathbf{p} \cdot \mathbf{g}_i}{\|\mathbf{p}\| \, \|\mathbf{g}_i\|}$$

Since ArcFace outputs are L2-normalized (‖**p**‖ = ‖**g**ᵢ‖ = 1), this simplifies to a dot product:

$$\text{sim}(\mathbf{p}, \mathbf{g}_i) = \mathbf{p} \cdot \mathbf{g}_i$$

### Decision Boundaries

$$
D(\mathbf{p}) = \begin{cases}
\texttt{FAMILY}_i & \text{if } \max_i \text{sim}(\mathbf{p}, \mathbf{g}_i) \geq \tau_{\text{accept}} = 0.45 \\
\texttt{UNCERTAIN} & \text{if } \tau_{\text{reject}} \leq \max_i \text{sim}(\mathbf{p}, \mathbf{g}_i) < \tau_{\text{accept}} \\
\texttt{STRANGER} & \text{if } \max_i \text{sim}(\mathbf{p}, \mathbf{g}_i) < \tau_{\text{reject}} = 0.35
\end{cases}
$$

### Temporal Consistency

For track $T_k$, an alert fires only after $N = 5$ consecutive stranger classifications:

$$
\text{Alert}(T_k) = \begin{cases}
\texttt{TRUE} & \text{if } \sum_{t=t_0}^{t_0+N-1} \mathbb{1}[D(\mathbf{p}_t^k) = \texttt{STRANGER}] \geq N \\
\texttt{FALSE} & \text{otherwise}
\end{cases}
$$

This eliminates single-frame false positives caused by bad angles, motion blur, or occlusion.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Runtime | Python 3.11+ | Core language |
| Inference | ONNX Runtime GPU 1.24+ | Model execution on CUDA |
| Detection | YOLOv11n (Ultralytics) | Person detection |
| Face Detection | SCRFD-10GF (InsightFace) | Face localization + landmarks |
| Recognition | ArcFace w600k_r50 (InsightFace) | 512-D face embeddings |
| Vector Search | FAISS (Facebook AI) | Embedding persistence + similarity |
| Computer Vision | OpenCV 4.13+ | Frame capture, preprocessing, display |
| Numerics | NumPy 2.x | Tensor operations, in-memory search |
| GPU | NVIDIA CUDA 12.x (pip packages) | GPU acceleration |

---

## Prerequisites

- **Python** 3.11 or newer
- **NVIDIA GPU** with 4+ GB VRAM (tested on GTX 1650)
- **NVIDIA Driver** 525+ (CUDA 12 compatible)
- **Windows 10/11** or Linux (Windows tested, Linux should work)
- **Webcam** or RTSP camera for live feed

> **Note:** No CUDA Toolkit installation required — CUDA libraries are installed automatically via pip.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Janmenjay30/HemidallEye.git
cd HemidallEye
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install CUDA Runtime Libraries (GPU acceleration)

```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12
```

> These provide the CUDA DLLs needed by `onnxruntime-gpu` without requiring a system-wide CUDA Toolkit installation.

---

## Model Download

Download the three required ONNX models (run once):

```bash
python -m heimdall.download_models
```

This downloads:

| Model | Size | Source |
|-------|------|--------|
| `yolo11n.onnx` | 11 MB | [Ultralytics assets](https://github.com/ultralytics/assets) |
| `det_10g.onnx` | 17 MB | [InsightFace buffalo_l](https://github.com/deepinsight/insightface/releases) |
| `w600k_r50.onnx` | 174 MB | [InsightFace buffalo_l](https://github.com/deepinsight/insightface/releases) |

Models are saved to the `models/` directory (git-ignored).

---

## Usage

### 1. Enroll Family Members

You need to register known faces **before** running the system. Multiple methods available:

#### From Webcam (Interactive)

```bash
python -m heimdall.enroll --name "Alice" --webcam --count 5
```

A preview window opens — press **SPACE** to capture, **ESC** to cancel. Tilt your head slightly between shots for better coverage.

#### From a Single Photo

```bash
python -m heimdall.enroll --name "Bob" --image ./photos/bob.jpg
```

#### From a Folder of Photos

```bash
python -m heimdall.enroll --name "Charlie" --images ./photos/charlie/
```

#### List Enrolled Persons

```bash
python -m heimdall.enroll --list
```

> **Tip:** Enroll **5–10 embeddings per person** with varied angles and lighting for best accuracy.

### 2. Run the System

#### Webcam

```bash
python -m heimdall.main --source 0
```

#### RTSP Camera

```bash
python -m heimdall.main --source rtsp://192.168.1.100:554/stream
```

#### Video File

```bash
python -m heimdall.main --source ./test_video.mp4
```

#### Headless Mode (No GUI)

```bash
python -m heimdall.main --source 0 --headless
```

Press **q** in the video window to quit (GUI mode), or **Ctrl+C** for headless.

### CLI Reference

```
heimdall.main
  --source SOURCE      RTSP URL, webcam index (0), or video file path (default: 0)
  --headless           Run without GUI display
  --log-level LEVEL    DEBUG | INFO | WARNING | ERROR (default: INFO)

heimdall.enroll
  --name NAME          Person's name to enroll
  --image PATH         Single face image path
  --images DIR         Directory containing face images
  --webcam             Enroll from webcam
  --count N            Number of webcam captures (default: 5)
  --list               List enrolled persons and exit
```

---

## Configuration

All parameters are centralized in [`heimdall/config.py`](heimdall/config.py). Key tunable values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_input_size` | 640 | YOLO input resolution (N×N) |
| `yolo_conf_threshold` | 0.45 | Min person detection confidence |
| `scrfd_conf_threshold` | 0.50 | Min face detection confidence |
| `tau_accept` | 0.45 | Similarity threshold for FAMILY |
| `tau_reject` | 0.35 | Similarity threshold for STRANGER |
| `stranger_consecutive_frames` | 5 | Frames before stranger alert fires |
| `target_fps` | 15 | Minimum acceptable FPS |
| `frame_skip` | 2 | Base frame skip (adaptive) |
| `min_face_size_px` | 40 | Ignore faces smaller than 40×40 |
| `laplacian_blur_threshold` | 50.0 | Anti-spoof blur detection threshold |
| `alert_cooldown_sec` | 30.0 | Seconds between alerts per track |
| `max_gallery_per_person` | 10 | Max embeddings stored per identity |

---

## Database Strategy

For a small family (≤ 50 people, ≤ 500 embeddings):

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Hot** | In-memory NumPy array | Instant dot-product similarity (< 0.01 ms) |
| **Persistent** | FAISS `IndexFlatIP` | Disk-backed inner product index |
| **Metadata** | JSON file | Name ↔ embedding ID mapping |

**Why not a full vector DB?** Overkill for < 1,000 vectors. NumPy dot product on 500 vectors takes < 0.01 ms.

Database files (in `data/`):

- `family_db.faiss` — FAISS index with all embeddings
- `family_meta.json` — Maps embedding IDs to person names

---

## Project Structure

```
Heimdall/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignores models/, data/, logs/, .venv/
│
├── heimdall/                  # Main package
│   ├── __init__.py
│   ├── config.py              # Centralized configuration (all tunable params)
│   ├── vram_manager.py        # GPU memory monitoring, ONNX session builder, CUDA DLL registration
│   ├── detector.py            # YOLOv11n person detection (letterbox + NMS)
│   ├── face_detector.py       # SCRFD-10GF face detection + 5-point alignment
│   ├── recognizer.py          # ArcFace w600k_r50 embedding extraction
│   ├── face_database.py       # In-memory NumPy + FAISS persistence
│   ├── decision_engine.py     # Similarity thresholds + temporal consistency
│   ├── tracker.py             # IoU-based multi-object tracker
│   ├── anti_spoof.py          # Laplacian variance + aspect ratio liveness checks
│   ├── pipeline.py            # Main orchestrator (ties all stages together)
│   ├── enroll.py              # Family face enrollment CLI
│   ├── main.py                # Entry point (webcam / RTSP / file + overlay)
│   └── download_models.py     # One-click model downloader
│
├── models/                    # ONNX model files (git-ignored)
│   ├── yolo11n.onnx
│   ├── det_10g.onnx
│   └── w600k_r50.onnx
│
├── data/                      # Face database (git-ignored)
│   ├── family_db.faiss
│   └── family_meta.json
│
├── logs/                      # Stranger event snapshots (git-ignored)
└── photos/                    # Enrollment source photos (git-ignored)
```

---

## Performance Benchmarks

Tested on **NVIDIA GTX 1650 (4 GB VRAM)**, Python 3.11, ONNX Runtime GPU 1.24:

| Metric | Value |
|--------|-------|
| **End-to-end FPS** | 25–30 FPS |
| **Person detection** | ~8 ms/frame |
| **Face detection** | ~3 ms/crop |
| **Face recognition** | ~5 ms/face |
| **Total per-frame** | ~36 ms (1 person) |
| **Peak VRAM** | ~2.6 GB |
| **VRAM headroom** | ~1.4 GB |
| **Database lookup** | < 0.01 ms (500 embeddings) |
| **Anti-spoof check** | < 1 ms/face (CPU-only) |

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: onnxruntime` | Activate the virtual environment first |
| `Could not locate nvrtc64_120_0.dll` | `pip install nvidia-cuda-nvrtc-cu12` |
| `Could not locate cublasLt64_12.dll` | `pip install nvidia-cublas-cu12` |
| Models fall back to CPU | Install all CUDA pip packages (see Installation step 4) |
| Webcam not opening | Ensure no other app is using the camera |
| `insightface` install fails | Not needed — we use ONNX models directly via onnxruntime |
| Low FPS | Increase `frame_skip` in config.py or lower `yolo_input_size` to 480 |

### Verify GPU is Active

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include: 'CUDAExecutionProvider'
```

---

## Roadmap

- [ ] Push notifications (Telegram / Pushbullet) for stranger alerts
- [ ] Web dashboard — live stream + event log in browser
- [ ] Multi-camera support — process multiple RTSP streams
- [ ] Dedicated anti-spoof DNN model for advanced liveness detection
- [ ] Auto-enrollment — learn new family members with user confirmation
- [ ] Event video recording — save clips when strangers are detected
- [ ] Docker deployment with NVIDIA Container Toolkit
- [ ] Edge optimization for NVIDIA Jetson Nano / Orin

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is open source under the [MIT License](LICENSE).

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11 object detection
- [InsightFace](https://github.com/deepinsight/insightface) — SCRFD face detection + ArcFace recognition
- [ONNX Runtime](https://onnxruntime.ai/) — Cross-platform GPU inference engine
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search (Facebook AI)
- [OpenCV](https://opencv.org/) — Computer vision primitives

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Janmenjay30">Janmenjay</a>
</p>
