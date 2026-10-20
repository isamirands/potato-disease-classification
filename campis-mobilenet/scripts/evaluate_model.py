import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import torch
from torch.utils.data import DataLoader
from campis.datasets import ManifestDataset
from campis.datasets.loader import collate
from campis.inference.preprocess import inference_transform
from campis.models import MultiCropMobileNet
from campis.training.checkpoint import load_checkpoint
from campis.evaluation.evaluate import evaluate
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--manifest",default="data/manifests/manifest.csv");p.add_argument("--output-dir",default="artifacts/reports");a=p.parse_args(); state=torch.load(a.checkpoint,map_location="cpu",weights_only=False); crops=state["crops"]; m=MultiCropMobileNet(crops,pretrained=False);load_checkpoint(a.checkpoint,m); device="cuda" if torch.cuda.is_available() else "cpu"; m.to(device); ds=ManifestDataset(a.manifest,crops,"test",inference_transform());print(f"[CAMPIS] Evaluando {len(ds)} imágenes en {device}...");print(evaluate(m,DataLoader(ds,batch_size=32,collate_fn=collate),crops,a.output_dir,device))
