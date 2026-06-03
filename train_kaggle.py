"""Kaggle-GPU training entry point for Pix2Vox on the satellite dataset.

On Kaggle, upload (as a Kaggle Dataset) the contents of `satellite_dataset/` and the patched
`Pix2Vox/` source tree, then run this script. It assumes a CUDA-capable GPU.

Run from inside the Pix2Vox directory:
    python train_kaggle.py --data_root /kaggle/input/satellite-dataset --epochs 150
"""
import argparse, os, sys, multiprocessing as mp, logging
import matplotlib; matplotlib.use("Agg")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from core.train import train_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Folder containing Satellite.json + per-satellite subfolders")
    p.add_argument("--out", default="/kaggle/working/output")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_views", type=int, default=4, help="N views per sample (1..24)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr_milestone", type=int, default=100)
    args = p.parse_args()

    cfg.DATASETS.SATELLITE.TAXONOMY_FILE_PATH = os.path.join(args.data_root, "Satellite.json")
    cfg.DATASETS.SATELLITE.RENDERING_PATH = os.path.join(args.data_root, "%s/rendering/%02d.png")
    cfg.DATASETS.SATELLITE.VOXEL_PATH = os.path.join(args.data_root, "%s/model.binvox")
    cfg.DATASET.TRAIN_DATASET = "Satellite"
    cfg.DATASET.TEST_DATASET = "Satellite"

    cfg.CONST.BATCH_SIZE = args.batch_size
    cfg.CONST.N_VIEWS_RENDERING = args.n_views
    cfg.TRAIN.NUM_WORKER = args.num_workers
    cfg.TRAIN.NUM_EPOCHES = args.epochs
    cfg.TRAIN.ENCODER_LR_MILESTONES = [args.lr_milestone]
    cfg.TRAIN.DECODER_LR_MILESTONES = [args.lr_milestone]
    cfg.TRAIN.REFINER_LR_MILESTONES = [args.lr_milestone]
    cfg.TRAIN.MERGER_LR_MILESTONES = [args.lr_milestone]
    cfg.TRAIN.SAVE_FREQ = 10
    cfg.TRAIN.EPOCH_START_USE_MERGER = 0
    cfg.TRAIN.EPOCH_START_USE_REFINER = 5

    cfg.DIR.OUT_PATH = args.out
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(0); torch.manual_seed(0)
    mp.log_to_stderr(); logging.getLogger().setLevel(logging.INFO)
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
          "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    train_net(cfg)


if __name__ == "__main__":
    main()
