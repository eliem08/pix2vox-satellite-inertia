"""Max-quality CPU training for the satellite dataset.

Differences vs local_train.py (the quick proof-of-concept):
  * multi-view: up to 5 of the 24 available views per object, view-count randomised
    per epoch (UPDATE_N_VIEWS_RENDERING) so the Merger learns to fuse and the model
    is robust to 1..5 views at inference;
  * 200 epochs (was 30) with a longer LR schedule;
  * refiner active from epoch 0.
Writes to ./output_better so the original ./output_local model is preserved.
Best checkpoint is saved whenever validation IoU improves, so progress is never lost.
"""
import os, sys, multiprocessing as mp, logging
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from core.train import train_net

cfg.DATASET.TRAIN_DATASET = "Satellite"
cfg.DATASET.TEST_DATASET = "Satellite"
cfg.CONST.BATCH_SIZE = 1
cfg.CONST.N_VIEWS_RENDERING = 5            # use up to 5 of the 24 views per object
cfg.TRAIN.UPDATE_N_VIEWS_RENDERING = True  # randomise 1..5 views each epoch (robust merger)
cfg.TRAIN.NUM_WORKER = 0                    # Windows-safe
cfg.TRAIN.NUM_EPOCHES = int(os.environ.get("EPOCHS", "200"))
cfg.TRAIN.VAL_FREQ = int(os.environ.get("VAL_FREQ", "20"))   # validate every 20 epochs
cfg.TRAIN.SAVE_FREQ = int(os.environ.get("SAVE_FREQ", "20"))
# keep photometric augmentation off (matches the run that produced a working model),
# but RandomBackground/flip/permute in core/train.py still provide variety
cfg.TRAIN.BRIGHTNESS = 0; cfg.TRAIN.CONTRAST = 0; cfg.TRAIN.SATURATION = 0; cfg.TRAIN.NOISE_STD = 0
cfg.TRAIN.EPOCH_START_USE_MERGER = 0
cfg.TRAIN.EPOCH_START_USE_REFINER = 0
cfg.TRAIN.ENCODER_LR_MILESTONES = [150, 220]
cfg.TRAIN.DECODER_LR_MILESTONES = [150, 220]
cfg.TRAIN.REFINER_LR_MILESTONES = [150, 220]
cfg.TRAIN.MERGER_LR_MILESTONES = [150, 220]
cfg.CONST.IMG_W = cfg.CONST.IMG_H = 224
cfg.CONST.CROP_IMG_W = cfg.CONST.CROP_IMG_H = 224
cfg.DIR.OUT_PATH = os.environ.get("OUTDIR", "./output_better")

np.random.seed(0); torch.manual_seed(0)
mp.log_to_stderr(); logging.getLogger().setLevel(logging.WARNING)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Multi-view training: up to", cfg.CONST.N_VIEWS_RENDERING, "views, ", cfg.TRAIN.NUM_EPOCHES, "epochs")
train_net(cfg)
print("\n[IMPROVED TRAIN DONE]")
