# Pix2Vox for Satellites — 3-D Shape Reconstruction & Moment-of-Inertia Estimation

Reconstruct a satellite's **3-D shape** from several **2-D images**, then estimate its
**moment-of-inertia tensor** from the reconstructed shape.

This is the [Pix2Vox](https://github.com/hzxie/Pix2Vox) architecture (Xie *et al.*, ICCV 2019)
adapted to a custom satellite dataset, with an added inference + physics pipeline.

```
2-D photos  ──►  Pix2Vox (deep net)  ──►  32×32×32 voxel model  ──►  moment of inertia
```

---

## How the model works

Four small networks in a chain (see `models/`):

| Stage | File | Role | Output |
|-------|------|------|--------|
| **Encoder** | `models/encoder.py` | Pretrained VGG16 turns each 224×224 image into features | 256×8×8 |
| **Decoder** | `models/decoder.py` | 3-D transposed convolutions → coarse volume (Sigmoid → probabilities) | 32³ |
| **Merger**  | `models/merger.py`  | Learned weighted fusion of the multiple views | 32³ |
| **Refiner** | `models/refiner.py` | 3-D U-Net-style clean-up | 32³ |

Output is a **32×32×32 grid of occupancy probabilities**. Training uses Binary Cross-Entropy
against the ground-truth `.binvox` shapes; quality is measured with IoU.

---

## Dataset layout

Kept **outside** the repo by default (`../satellite_dataset/`, git-ignored):

```
satellite_dataset/
├── Satellite.json              # train / val / test split + taxonomy
├── Aqua/
│   ├── rendering/00.png …      # 2-D input views
│   └── model.binvox            # ground-truth 3-D voxel model
├── Calipso/ …
└── …
```

Split (`Satellite.json`): **train** = Aqua, Calipso, CloudSat, ICECube, ICESat2, LRO, MiRaTA ·
**val** = Sentinel6 · **test** = SolarB.

---

## Setup

```bash
# 1) PyTorch CPU build (matched pair — do this first)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu
# 2) the rest
pip install -r requirements.txt
```

> **Environment gotcha (important):** this stack requires **NumPy < 2**. A stray NumPy 2.x
> (e.g. in a per-user site-packages) will break OpenCV and torchvision with errors like
> `numpy.core.multiarray failed to import` or `operator torchvision::nms does not exist`.
> Use a clean virtual environment with the pinned versions above.

---

## Usage

```bash
# Train on CPU (~30 epochs; writes checkpoints to output_local/)
python local_train.py

# Quick end-to-end smoke test (2 epochs)
python smoke_test.py

# Single object: reconstruct + moment of inertia
python infer_and_inertia.py --weights output_local/checkpoints/<run>/best-ckpt.pth \
                            --img_dir ../satellite_dataset/Aqua/rendering --mass 2900

# All satellites: reconstruct + inertia + comparison figures (used for the report)
python batch_infer_inertia.py        # → report_assets/{results.json, *_card.png}
```

---

## Moment of inertia

For each reconstructed shape (`inertia_from_voxels` in `batch_infer_inertia.py`):

1. Keep voxels above a probability threshold; treat each as a small cube at its centre.
2. Spread the total mass uniformly over the solid voxels (uniform-density assumption).
3. Compute the centre of mass.
4. Build the 3×3 inertia tensor about the COM (diagonal = moments, off-diagonal = products).
5. Diagonalise → **principal moments** (eigenvalues) and **principal axes** (eigenvectors).

Inertia scales linearly with the assumed mass and with the square of the voxel size, so plug in
the real mass/scale to get physical magnitudes.

---

## Results (this proof-of-concept run, 30 CPU epochs)

| Satellite | Split | Pred fill | GT fill | IoU |
|-----------|-------|-----------|---------|-----|
| Aqua | train | 4.4% | 4.4% | 0.098 |
| CloudSat | train | 6.1% | 4.4% | 0.151 |
| ICECube | train | 6.2% | 4.4% | 0.165 |
| ICESat2 | train | 5.5% | 3.5% | 0.158 |
| Sentinel6 | val | 6.7% | 6.8% | 0.128 |
| SolarB | test | 4.9% | 1.3% | 0.003 |

The model learns the rough *amount* of mass; shape fidelity (IoU) is modest at this small
training budget and improves with more data, GPU training, and more epochs. A full write-up with
figures and per-satellite inertia tensors is in **`Pix2Vox_Teaching_Report.pdf`**.

---

## Project structure (key files)

```
config.py                 # all settings (image size, 32³ grid, LR, dataset paths)
models/{encoder,decoder,merger,refiner}.py
core/{train,test}.py      # training loop & evaluation (IoU)
utils/data_loaders.py     # incl. SatelliteDataLoader
utils/binvox_rw.py        # read/write .binvox
local_train.py            # CPU training entry point
infer_and_inertia.py      # single-object inference + inertia
batch_infer_inertia.py    # all-satellites inference + inertia + figures
```

---

## Credits & license

Built on **Pix2Vox** by Haozhe Xie *et al.* — original repository:
<https://github.com/hzxie/Pix2Vox> (MIT license, see `LICENSE`). Satellite-dataset adaptation,
the inference/inertia pipeline, and the reporting tools were added on top.

```bibtex
@inproceedings{xie2019pix2vox,
  title={Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images},
  author={Xie, Haozhe and Yao, Hongxun and Sun, Xiaoshuai and Zhou, Shangchen and Zhang, Shengping},
  booktitle={ICCV},
  year={2019}
}
```
