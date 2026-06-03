"""Run a trained Pix2Vox checkpoint on a folder of RGB images and compute
the moment-of-inertia tensor of the predicted voxel grid.

Usage:
    python infer_and_inertia.py --weights output/checkpoints/.../ckpt-best.pth \
                                 --img_dir ../satellite_dataset/Aqua/rendering \
                                 --mass 1500
"""
import os, sys, argparse
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import torchvision.transforms as T

from config import cfg
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger


def load_pix2vox(weights_path, device):
    encoder = Encoder(cfg).to(device)
    decoder = Decoder(cfg).to(device)
    refiner = Refiner(cfg).to(device)
    merger = Merger(cfg).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)  # our own checkpoint, safe
    def strip(sd):  # state_dict was saved from DataParallel? strip 'module.' prefix
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    encoder.load_state_dict(strip(ckpt["encoder_state_dict"]))
    decoder.load_state_dict(strip(ckpt["decoder_state_dict"]))
    refiner.load_state_dict(strip(ckpt["refiner_state_dict"]))
    merger.load_state_dict(strip(ckpt["merger_state_dict"]))
    for m in (encoder, decoder, refiner, merger):
        m.eval()
    return encoder, decoder, refiner, merger


def load_images(img_dir, n_views, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    paths = sorted([os.path.join(img_dir, n) for n in os.listdir(img_dir) if n.lower().endswith(".png")])
    paths = paths[:n_views]
    norm = T.Compose([T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))])
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (cfg.CONST.IMG_W, cfg.CONST.IMG_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(norm(img))
    return torch.stack(imgs, dim=0).unsqueeze(0)  # (1, N, 3, H, W)


def predict_volume(images, models):
    encoder, decoder, refiner, merger = models
    with torch.no_grad():
        feats = encoder(images)
        raw, vols = decoder(feats)
        vols = merger(raw, vols)
        vols = refiner(vols)
    return vols.squeeze(0).cpu().numpy()  # (D, H, W)


def inertia_from_occupancy(occ, voxel_size_m=1.0, total_mass=1.0, threshold=0.4):
    """occ: (D,H,W) probability. Returns 3x3 inertia tensor + principal moments + COM."""
    occ_bin = occ >= threshold
    idx = np.argwhere(occ_bin)  # (M, 3)
    if len(idx) == 0:
        raise RuntimeError("Predicted volume is empty at threshold %s" % threshold)
    pts = (idx + 0.5) * voxel_size_m
    com = pts.mean(0)
    rel = pts - com
    m = total_mass / len(pts)
    r2 = (rel * rel).sum(1)
    I = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            if a == b:
                I[a, b] = (m * (r2 - rel[:, a] * rel[:, b])).sum()
            else:
                I[a, b] = -(m * rel[:, a] * rel[:, b]).sum()
    I += np.eye(3) * (m * len(pts) * voxel_size_m**2 / 6.0)  # self-inertia of cube voxels
    eigvals, eigvecs = np.linalg.eigh(I)
    return I, eigvals, eigvecs, com, len(pts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--voxel_size", type=float, default=1.0, help="voxel edge in metres (depends on satellite scale)")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--save_pred", default=None, help="optional path to save predicted volume as .npy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    models = load_pix2vox(args.weights, device)
    images = load_images(args.img_dir, args.n_views).to(device)
    vol = predict_volume(images, models)
    print(f"predicted volume shape: {vol.shape}   fill@{args.threshold}: {(vol >= args.threshold).mean():.3f}")
    if args.save_pred:
        np.save(args.save_pred, vol)
        print("saved", args.save_pred)

    I, evals, evecs, com, n = inertia_from_occupancy(vol, args.voxel_size, args.mass, args.threshold)
    print("\n=== Moment of inertia ===")
    print(f"occupied voxels   : {n} / {vol.size}")
    print(f"COM (voxel coords): {com}")
    print(f"inertia tensor    :\n{I}")
    print(f"principal moments : {evals}")
    print(f"principal axes    :\n{evecs}")
