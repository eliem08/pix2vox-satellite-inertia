"""Real local training run on CPU. ~30 epochs, validation every 10."""
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
cfg.CONST.N_VIEWS_RENDERING = 1
cfg.TRAIN.NUM_WORKER = 0
cfg.TRAIN.NUM_EPOCHES = 30
cfg.TRAIN.VAL_FREQ = 10                # validate every 10 epochs (saves time)
cfg.TRAIN.SAVE_FREQ = 10
cfg.TRAIN.BRIGHTNESS = 0; cfg.TRAIN.CONTRAST = 0; cfg.TRAIN.SATURATION = 0; cfg.TRAIN.NOISE_STD = 0
cfg.TRAIN.EPOCH_START_USE_MERGER = 0
cfg.TRAIN.EPOCH_START_USE_REFINER = 3
cfg.TRAIN.ENCODER_LR_MILESTONES = [20]
cfg.TRAIN.DECODER_LR_MILESTONES = [20]
cfg.TRAIN.REFINER_LR_MILESTONES = [20]
cfg.TRAIN.MERGER_LR_MILESTONES = [20]
cfg.CONST.IMG_W = cfg.CONST.IMG_H = 224
cfg.CONST.CROP_IMG_W = cfg.CONST.CROP_IMG_H = 224
cfg.DIR.OUT_PATH = "./output_local"

np.random.seed(0); torch.manual_seed(0)
mp.log_to_stderr(); logging.getLogger().setLevel(logging.WARNING)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
train_net(cfg)
print("\n[LOCAL TRAIN DONE]")
