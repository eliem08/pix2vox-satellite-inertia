"""Quick CPU smoke test for Pix2Vox on the satellite dataset.

Trains 2 epochs at batch_size=1 with num_workers=0 (Windows-friendly) just to verify
the whole pipeline (loaders -> encoder -> decoder -> merger -> refiner -> loss -> backprop)
runs end-to-end.

Run from the Pix2Vox directory.
"""
import os, sys, multiprocessing as mp, logging
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from core.train import train_net

# ---- overrides for the smoke test ----
cfg.DATASET.TRAIN_DATASET = "Satellite"
cfg.DATASET.TEST_DATASET = "Satellite"
cfg.CONST.BATCH_SIZE = 1            # we only have 7 train samples
cfg.CONST.N_VIEWS_RENDERING = 1     # single view per sample for the smoke test
cfg.TRAIN.NUM_WORKER = 0            # Windows-safe
cfg.TRAIN.NUM_EPOCHES = 2
cfg.TRAIN.BRIGHTNESS = 0; cfg.TRAIN.CONTRAST = 0; cfg.TRAIN.SATURATION = 0; cfg.TRAIN.NOISE_STD = 0
cfg.TRAIN.SAVE_FREQ = 1
cfg.TRAIN.EPOCH_START_USE_MERGER = 0
cfg.TRAIN.EPOCH_START_USE_REFINER = 1   # postpone refiner so first epoch is fast

cfg.CONST.IMG_W = 224; cfg.CONST.IMG_H = 224
cfg.CONST.CROP_IMG_W = 224; cfg.CONST.CROP_IMG_H = 224  # no random crop downsizing
cfg.DIR.OUT_PATH = "./output_smoke"

np.random.seed(0); torch.manual_seed(0)
mp.log_to_stderr(); logging.getLogger().setLevel(logging.INFO)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Train dataset:", cfg.DATASET.TRAIN_DATASET)
print("Voxel template:", cfg.DATASETS.SATELLITE.VOXEL_PATH)
print("Image template:", cfg.DATASETS.SATELLITE.RENDERING_PATH)
print("Tax file:", cfg.DATASETS.SATELLITE.TAXONOMY_FILE_PATH)

train_net(cfg)
print("\n[SMOKE TEST OK]")
