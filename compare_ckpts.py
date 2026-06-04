import os, sys
PROJ = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJ, ".torch_cache"))
sys.path.insert(0, PROJ)
import numpy as np, cv2, torch
import torchvision.transforms as T
from config import cfg
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
import utils.binvox_rw

DS = os.path.normpath(os.path.join(PROJ, "..", "satellite_dataset"))
CKPTS = {
    "old-30ep-1v":  os.path.join(PROJ, "output_local/checkpoints/2026-05-24T22-00-34/best-ckpt.pth"),
    "new-ep30-mv":  os.path.join(PROJ, "output_better/checkpoints/2026-06-04T16-16-29/best-ckpt.pth"),
    "new-ep60-mv":  os.path.join(PROJ, "output_better/checkpoints/2026-06-04T16-16-29/ckpt-epoch-0060.pth"),
    "new-ep120-mv": os.path.join(PROJ, "output_better/checkpoints/2026-06-04T16-16-29/ckpt-epoch-0120.pth"),
}
device = torch.device("cpu")

def load_model(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    strip = lambda sd: {k.replace("module.","",1):v for k,v in sd.items()}
    e,d,r,m = Encoder(cfg),Decoder(cfg),Refiner(cfg),Merger(cfg)
    e.load_state_dict(strip(ck["encoder_state_dict"]))
    d.load_state_dict(strip(ck["decoder_state_dict"]))
    r.load_state_dict(strip(ck["refiner_state_dict"]))
    m.load_state_dict(strip(ck["merger_state_dict"]))
    for x in (e,d,r,m): x.eval()
    return e,d,r,m

norm = T.Compose([T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])

def load_imgs(img_dir, n=4):
    paths = sorted(p for p in os.listdir(img_dir) if p.lower().endswith(".png"))[:n]
    xs = []
    for p in paths:
        im = cv2.imread(os.path.join(img_dir,p), cv2.IMREAD_COLOR)
        im = cv2.resize(im,(224,224))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        xs.append(norm(im))
    return torch.stack(xs,0).unsqueeze(0)

@torch.no_grad()
def predict(models, imgs):
    e,d,r,m = models
    f=e(imgs); raw,v=d(f); v=m(raw,v); v=r(v)
    return v.squeeze(0).cpu().numpy()

def best_iou(vol, gt):
    best_t, best_i = 0.2, 0.0
    for t in np.linspace(0.05, 0.70, 66):
        p = vol >= t
        i_ = (p & gt).sum(); u_ = (p | gt).sum()
        iou = float(i_)/float(u_) if u_ else 0.0
        if iou > best_i: best_i = iou; best_t = t
    return best_t, best_i

sats = sorted(d for d in os.listdir(DS) if os.path.isdir(os.path.join(DS,d)))
print("%-12s" % "satellite", "  ".join("%-22s" % k for k in CKPTS))
print("-" * 105)
for sat in sats:
    img_dir = os.path.join(DS, sat, "rendering")
    binv = os.path.join(DS, sat, "model.binvox")
    if not (os.path.isdir(img_dir) and os.path.exists(binv)): continue
    imgs = load_imgs(img_dir, n=4)
    with open(binv,"rb") as f: gt = utils.binvox_rw.read_as_3d_array(f).data.astype(bool)
    row = ["%-12s" % sat]
    for name, path in CKPTS.items():
        models = load_model(path)
        vol = predict(models, imgs)
        t, iou = best_iou(vol, gt)
        row.append("iou=%.3f t=%.2f max=%.2f" % (iou, t, vol.max()))
    print("  ".join(row))
