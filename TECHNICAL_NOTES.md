# Technical Notes — Ground Truth, Results & Roadmap

> This document addresses key technical questions about the methodology,
> current results, and the path to production-quality reconstruction.

---

## 1. How was the ground truth established?

The ground truth 3-D shapes were derived from existing **CAD (Computer-Aided Design)
models** of each satellite — the precise engineering geometry used during the
satellites' design and construction.

Each CAD model was converted into a **32×32×32 voxel grid** (a `.binvox` file),
which discretises the continuous 3-D shape into a grid of 32,768 small cubes.
This is the standard benchmark format used in 3-D reconstruction research and
matches exactly the format the Pix2Vox algorithm is trained and evaluated against.

---

## 2. Why does the predicted shape not yet closely resemble the input image?

The interpretation is accurate — the discrepancy is real, and has three root causes:

### a) Computational constraints
The model was trained on a **CPU only**. A full-scale training run typically
requires a dedicated GPU and several hundred epochs over many hours. Here we
were limited to 120 epochs — sufficient to prove the pipeline, but not to achieve
sharp, detail-rich reconstructions.

### b) Very small training set
The model was trained on only **7 satellite objects**. Deep learning models of
this type typically require hundreds to thousands of diverse 3-D shapes to
generalise well. With 7 objects, the model learns a rough approximation of
*"what a satellite looks like in general"* rather than the precise geometry of
each individual one.

### c) Voxel grid resolution
The output is a **32×32×32 grid**, so the finest detail the model can represent
is one 32nd of the object's bounding box. Fine structural features — solar panel
arrays, antenna booms, instrument mounts — are smaller than one voxel and are
therefore inherently unresolvable at this resolution.

---

## 3. Would GPU training and a larger dataset significantly improve results?

**Yes — emphatically.** The improvement would be substantial across all three dimensions:

| Upgrade | Expected impact |
|---|---|
| GPU training | Allows 500–1,000 epochs in the same wall-clock time as our 120 CPU epochs |
| More training objects | Hundreds of satellite CAD models are publicly available; the model learns structural patterns (solar panels, cylindrical bodies, appendages) that generalise to unseen satellites |
| Finer grid (64³ or 128³) | Preserves fine structural detail currently lost at 32³ |

As a concrete reference: the original Pix2Vox paper, trained on the full
ShapeNet dataset (~50,000 objects, GPU cluster), achieves **IoU scores of
0.45–0.65** depending on object category — roughly **3–5× higher** than what
was achieved here under the current constraints.

---

## 4. Is this a proof-of-concept?

**Yes, and that is precisely how it should be understood.** What has been demonstrated:

- The **complete pipeline is implemented and functional**:
  2-D images → 3-D voxel reconstruction → moment-of-inertia tensor computation.
- The **physics output is consistent and correct**: the inertia tensor calculation
  is standard rigid-body mechanics, independent of model fidelity, and will scale
  correctly once the 3-D shapes improve.
- The model **already shows measurable learning**: after the improved training run
  (120 epochs, multi-view), mean IoU improved **+9%** and maximum voxel confidence
  increased from **0.55 → 0.74** — predictions became compact solid masses rather
  than scattered clouds.
- The architecture, satellite dataset integration, and training infrastructure are
  **all in place and ready to scale**.

Think of it as a car engine running on a test bench.
The engineering is sound; what it needs now is more fuel (data) and time on a
proper track (GPU).

---

## 5. Improvement achieved in this delivery (v1 → v2)

| Metric | v1 (30 ep, 1-view) | v2 (120 ep, multi-view) | Change |
|---|---|---|---|
| Refiner loss | ~8.4 | ~1.6 | **5× lower** |
| Max voxel confidence | ~0.55 | ~0.74 | **+35%** |
| Mean IoU (all satellites) | 0.118 | 0.129 | **+9%** |
| Aqua IoU | 0.098 | 0.126 | +28% |
| LRO IoU | 0.131 | 0.181 | +38% |
| Sentinel6 IoU (unseen val) | 0.128 | 0.155 | +21% |
| SolarB IoU (fully unseen test) | 0.003 | 0.012 | +292% |

The key visual change: predictions went from **diffuse scattered clouds of dots**
to **compact, solid, dense masses** — a clear qualitative improvement visible in
the before/after comparison figures in the teaching report.

---

## 6. What would the full-scale version look like?

The recommended path to production-quality results:

1. **GPU environment** (cloud or local) — NVIDIA A100 or equivalent
2. **Expanded dataset** — 200–500 satellite CAD models (ESA, NASA open repositories)
3. **250+ training epochs** with full data augmentation
4. **64³ or 128³ voxel grid** for structural detail
5. **Per-object fine-tuning** for specific satellites of interest

Under these conditions, reconstructions closely matching the input images
(IoU > 0.40) are a realistic and achievable target.

---

## Deliverables

| File | Description |
|---|---|
| `Pix2Vox_Teaching_Report.pdf` | Full 8-page technical & teaching report with before/after figures, inertia tensors, and code walkthrough |
| `pix2vox-satellite-handover.zip` | Complete runnable package: source code, trained model (120-epoch), satellite dataset, reports |
| This repository | Full source code, training scripts, inference pipeline |

Full technical documentation, the trained model weights, and the reproducible
pipeline are included in the handover package.
