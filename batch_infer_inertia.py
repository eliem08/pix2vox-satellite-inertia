"""
batch_infer_inertia.py
======================
Run the trained Pix2Vox model on every satellite in the dataset and, for each one:
  1. reconstruct a 3D voxel volume from its 2D rendered views,
  2. extract a solid shape by thresholding the predicted occupancy probabilities,
  3. compute the moment-of-inertia tensor of that shape,
  4. render a comparison figure (input view | predicted voxels | ground-truth voxels).
Finally write report_assets/results.json, used to build the teaching report.

Environment (matches the training stack):
  torch 2.2.1 + torchvision 0.17.1 + numpy<2 + opencv-python + matplotlib
Run:
  python batch_infer_inertia.py
"""
import os, sys, json, argparse
PROJ = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJ, ".torch_cache"))
sys.path.insert(0, PROJ)

import numpy as np
import cv2
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D projection)

from config import cfg
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
import utils.binvox_rw


# --------------------------------------------------------------------- model loading
def load_pix2vox(weights_path, device):
    enc, dec, ref, mrg = Encoder(cfg), Decoder(cfg), Refiner(cfg), Merger(cfg)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    strip = lambda sd: {k.replace("module.", "", 1): v for k, v in sd.items()}
    enc.load_state_dict(strip(ckpt["encoder_state_dict"]))
    dec.load_state_dict(strip(ckpt["decoder_state_dict"]))
    ref.load_state_dict(strip(ckpt["refiner_state_dict"]))
    mrg.load_state_dict(strip(ckpt["merger_state_dict"]))
    for m in (enc, dec, ref, mrg):
        m.to(device).eval()
    return enc, dec, ref, mrg


def load_images(img_dir, n_views):
    paths = sorted(p for p in os.listdir(img_dir) if p.lower().endswith(".png"))[:n_views]
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    xs = []
    for p in paths:
        im = cv2.imread(os.path.join(img_dir, p), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (cfg.CONST.IMG_W, cfg.CONST.IMG_H))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        xs.append(norm(im))
    return torch.stack(xs, 0).unsqueeze(0), paths


@torch.no_grad()
def predict_volume(models, images):
    enc, dec, ref, mrg = models
    feats = enc(images)            # 2D views -> per-view image features
    raw, vols = dec(feats)         # features -> coarse 3D volume per view
    vols = mrg(raw, vols)          # fuse the views into one volume
    vols = ref(vols)               # refine (clean up) the fused volume
    return vols.squeeze(0).cpu().numpy()   # (32,32,32) probabilities


# --------------------------------------------------------------------- physics
def inertia_from_voxels(occ_bool, voxel_size_m, total_mass_kg):
    """Moment-of-inertia tensor of a set of occupied cubic voxels (uniform density).

    Returns (I 3x3, principal_moments, principal_axes, centre_of_mass, n_voxels).
    """
    idx = np.argwhere(occ_bool)                 # (M,3) indices of occupied voxels
    pts = (idx + 0.5) * voxel_size_m            # voxel-centre coordinates [metres]
    m = total_mass_kg / len(pts)                # equal share of mass per voxel
    com = pts.mean(0)                           # centre of mass
    rel = pts - com                             # positions relative to the COM
    r2 = (rel * rel).sum(1)
    I = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            if a == b:                          # diagonal -> moments of inertia
                I[a, b] = (m * (r2 - rel[:, a] ** 2)).sum()
            else:                               # off-diagonal -> products of inertia
                I[a, b] = -(m * rel[:, a] * rel[:, b]).sum()
    # each voxel is a small solid cube, not a point: add its own (1/6)m s^2 self-inertia
    I += np.eye(3) * (total_mass_kg * voxel_size_m ** 2 / 6.0)
    evals, evecs = np.linalg.eigh(I)            # principal moments + principal axes
    return I, evals, evecs, com, len(pts)


def iou(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    return float(inter) / float(union) if union else 0.0


# --------------------------------------------------------------------- rendering
def _tidy3d(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=18, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render_card(out_png, name, split, input_img_path, pred_bool, gt_bool):
    fig = plt.figure(figsize=(11, 3.7))
    ax0 = fig.add_subplot(1, 3, 1)
    img = cv2.cvtColor(cv2.imread(input_img_path), cv2.COLOR_BGR2RGB)
    ax0.imshow(img); ax0.axis("off"); ax0.set_title("Input view (1 of N)", fontsize=10)

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1.voxels(pred_bool, facecolors="#3a78c2")
    ax1.set_title("Predicted 3D voxels", fontsize=10); _tidy3d(ax1)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    ax2.voxels(gt_bool, facecolors="#9aa0a6")
    ax2.set_title("Ground-truth 3D voxels", fontsize=10); _tidy3d(ax2)

    fig.suptitle(f"{name}   ({split})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(PROJ, "output_local", "checkpoints",
                                                      "2026-05-24T22-00-34", "best-ckpt.pth"))
    ap.add_argument("--dataset", default=os.path.normpath(os.path.join(PROJ, "..", "satellite_dataset")))
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--mass", type=float, default=1000.0, help="assumed total mass [kg] (inertia scales linearly)")
    ap.add_argument("--voxel_size", type=float, default=0.5, help="voxel edge length [m]")
    ap.add_argument("--out", default=os.path.join(PROJ, "report_assets"))
    ap.add_argument("--no_render", action="store_true", help="skip 3D figure rendering (reuse existing)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    models = load_pix2vox(args.weights, device)

    tax = json.load(open(os.path.join(args.dataset, "Satellite.json"), encoding="utf-8"))[0]
    split = {s: "train" for s in tax["train"]}
    split.update({s: "val" for s in tax["val"]})
    split.update({s: "test" for s in tax["test"]})
    order = tax["train"] + tax["val"] + tax["test"]

    results = []
    print(f"{'satellite':12s}{'split':6s}{'voxels':>7s}{'fillP':>7s}{'fillGT':>7s}{'IoU':>7s}   principal moments [kg m^2]")
    for name in order:
        sd = os.path.join(args.dataset, name)
        img_dir, binv = os.path.join(sd, "rendering"), os.path.join(sd, "model.binvox")
        if not (os.path.isdir(img_dir) and os.path.exists(binv)):
            continue
        images, paths = load_images(img_dir, args.n_views)
        vol = predict_volume(models, images.to(device))
        with open(binv, "rb") as fh:
            gt = utils.binvox_rw.read_as_3d_array(fh).data.astype(bool)

        pred = vol >= args.threshold
        if pred.sum() < 50:                      # robustness: never produce an empty shape
            k = max(int(gt.sum()), 100)
            pred = vol >= np.sort(vol.ravel())[-k]

        I, evals, evecs, com, n = inertia_from_voxels(pred, args.voxel_size, args.mass)
        Ig, evals_g, evecs_g, com_g, ng = inertia_from_voxels(gt, args.voxel_size, args.mass)
        card = os.path.join(args.out, f"{name}_card.png")
        if not args.no_render:
            render_card(card, name, split[name], os.path.join(img_dir, paths[0]), pred, gt)

        rec = dict(name=name, split=split[name], n_voxels=int(n),
                   fill_pred=float(pred.mean()), fill_gt=float(gt.mean()), iou=float(iou(pred, gt)),
                   max_prob=float(vol.max()), mean_prob=float(vol.mean()),
                   com=[round(float(x), 2) for x in com],
                   inertia=[[round(float(I[i, j]), 1) for j in range(3)] for i in range(3)],
                   principal_moments=[round(float(x), 1) for x in evals],
                   principal_axes=[[round(float(evecs[i, j]), 3) for j in range(3)] for i in range(3)],
                   n_voxels_gt=int(ng),
                   com_gt=[round(float(x), 2) for x in com_g],
                   inertia_gt=[[round(float(Ig[i, j]), 1) for j in range(3)] for i in range(3)],
                   principal_moments_gt=[round(float(x), 1) for x in evals_g],
                   card=os.path.basename(card))
        results.append(rec)
        pm = "  ".join(f"{v:,.0f}" for v in evals)
        print(f"{name:12s}{split[name]:6s}{n:7d}{pred.mean()*100:6.1f}%{gt.mean()*100:6.1f}%{rec['iou']:7.3f}   [{pm}]")

    out_json = os.path.join(args.out, "results.json")
    json.dump(dict(config=dict(threshold=args.threshold, mass=args.mass,
                               voxel_size=args.voxel_size, n_views=args.n_views),
                   results=results), open(out_json, "w"), indent=2)
    print("\nWrote", out_json, "and", len(results), "render cards to", args.out)


if __name__ == "__main__":
    main()
