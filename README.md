# Heimdall — Family vs. Stranger CCTV Recognition System

> *"The all-seeing guardian of your home."*

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RTSP Camera Stream                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ decode (OpenCV + FFMPEG hw accel)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Frame Acquisition & Preprocessing                       │
│  • Grab every Nth frame (adaptive skip based on GPU load)          │
│  • Resize to 640×640 for detection                                 │
│  • Color normalization (BGR → RGB, /255, float16)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Person Detection (YOLOv11n — ONNX, FP16)               │
│  • Detects "person" class only → crops bounding boxes              │
│  • Filters by confidence ≥ 0.45 and area ≥ 3000 px²               │
│  • NMS IoU threshold = 0.5                                         │
│  ~1.2 GB VRAM  |  ~8ms/frame on GTX 1650                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Face Detection & Alignment (SCRFD-2.5GF — ONNX, FP16)  │
│  • Runs ONLY inside person bounding boxes (saves compute)          │
│  • 5-point landmark alignment → 112×112 normalized face            │
│  • Filters faces < 40×40 px (too small for reliable recognition)   │
│  ~0.3 GB VRAM  |  ~3ms/crop                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Face Recognition (ArcFace-R50 — ONNX, FP16)            │
│  • Produces 512-D L2-normalized embedding                          │
│  • Compare against enrolled family DB via Cosine Similarity        │
│  ~0.5 GB VRAM  |  ~5ms/face                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Decision Engine                                         │
│  • Cosine Similarity thresholds:                                   │
│      FAMILY:   sim ≥ 0.45  (high recall)                          │
│      UNKNOWN:  sim < 0.35  (confirmed stranger)                   │
│      GREY:     0.35 ≤ sim < 0.45  (uncertain → require more frames│
│  • Temporal Consistency Filter:                                     │
│      Must see same unknown face in ≥ 5 consecutive frames          │
│      Uses track-ID from simple IoU tracker                         │
│  • Anti-Photo Spoof:                                               │
│      Laplacian variance check (blur ≤ 50 → flat/photo → reject)   │
│      Optional: aspect-ratio sanity check                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — Alert & Logging                                         │
│  • Log stranger event with timestamp + cropped face image          │
│  • Optional: push notification / webhook                           │
│  • All data stays LOCAL — zero cloud dependency                    │
└─────────────────────────────────────────────────────────────────────┘
```

## VRAM Budget (GTX 1650 — 4096 MB)

| Component              | Precision | VRAM (MB) | Notes                       |
|------------------------|-----------|-----------|-----------------------------|
| YOLOv11n (detection)   | FP16      | ~1200     | Nano variant, 640 input     |
| SCRFD-2.5GF (face det) | FP16      | ~300      | Lightweight face detector   |
| ArcFace-R50 (embed)    | FP16      | ~500      | ResNet-50 backbone          |
| ONNX Runtime overhead  | —         | ~400      | CUDA context, workspace     |
| Frame buffers          | —         | ~200      | IO tensors, pre/post proc   |
| **TOTAL**              |           | **~2600** | **1.4 GB headroom**        |

## Cosine Similarity — Mathematical Framework

Given a probe embedding $\mathbf{p} \in \mathbb{R}^{512}$ and gallery embedding $\mathbf{g}_i$:

$$\text{sim}(\mathbf{p}, \mathbf{g}_i) = \frac{\mathbf{p} \cdot \mathbf{g}_i}{\|\mathbf{p}\| \, \|\mathbf{g}_i\|}$$

Since ArcFace outputs are L2-normalized ($\|\mathbf{p}\| = \|\mathbf{g}_i\| = 1$), this simplifies to:

$$\text{sim}(\mathbf{p}, \mathbf{g}_i) = \mathbf{p} \cdot \mathbf{g}_i$$

**Decision boundaries:**

$$
D(\mathbf{p}) = \begin{cases}
\texttt{FAMILY}_i & \text{if } \max_i \text{sim}(\mathbf{p}, \mathbf{g}_i) \geq \tau_{\text{accept}} = 0.45 \\
\texttt{UNCERTAIN} & \text{if } \tau_{\text{reject}} \leq \max_i \text{sim}(\mathbf{p}, \mathbf{g}_i) < \tau_{\text{accept}} \\
\texttt{STRANGER} & \text{if } \max_i \text{sim}(\mathbf{p}, \mathbf{g}_i) < \tau_{\text{reject}} = 0.35
\end{cases}
$$

**Temporal consistency** for track $T_k$:

$$
\text{Alert}(T_k) = \begin{cases}
\texttt{TRUE} & \text{if } \sum_{t=t_0}^{t_0+N-1} \mathbb{1}[D(\mathbf{p}_t^k) = \texttt{STRANGER}] \geq N, \quad N=5 \\
\texttt{FALSE} & \text{otherwise}
\end{cases}
$$

## Database Strategy

For a small family (≤ 50 people, ≤ 500 embeddings with augmentations):

- **Primary store**: In-memory NumPy array — instant dot-product similarity
- **Persistent backup**: FAISS `IndexFlatIP` (Inner Product) saved to disk
- **Why not a full vector DB?** Overkill for < 1000 vectors; NumPy achieves < 0.01ms lookup

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (run once)
python heimdall/download_models.py

# 3. Enroll family faces
python heimdall/enroll.py --name "Alice" --images ./photos/alice/

# 4. Run the system
python heimdall/main.py --source rtsp://192.168.1.100:554/stream
```

## Project Structure

```
Heimdall/
├── README.md
├── requirements.txt
├── heimdall/
│   ├── __init__.py
│   ├── config.py              # All tunable parameters
│   ├── vram_manager.py        # GPU memory monitoring & budgeting
│   ├── detector.py            # YOLOv11n person detection
│   ├── face_detector.py       # SCRFD face detection + alignment
│   ├── recognizer.py          # ArcFace embedding extraction
│   ├── face_database.py       # FAISS + in-memory embedding store
│   ├── decision_engine.py     # Thresholding + temporal consistency
│   ├── tracker.py             # Simple IoU-based person tracker
│   ├── anti_spoof.py          # Liveness / anti-photo checks
│   ├── pipeline.py            # Main orchestrator
│   ├── enroll.py              # Family face enrollment CLI
│   ├── main.py                # Entry point
│   └── download_models.py     # Model downloader
├── models/                    # ONNX model files (git-ignored)
├── data/
│   ├── family_db.faiss        # Persisted FAISS index
│   └── family_meta.json       # Name ↔ embedding ID mapping
├── logs/                      # Stranger event logs + snapshots
└── photos/                    # Enrollment source photos
```
#   H e m i d a l l E y e  
 