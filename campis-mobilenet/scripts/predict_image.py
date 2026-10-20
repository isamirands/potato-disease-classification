import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from campis.inference import Predictor
p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="artifacts/checkpoints/best_model.pt");p.add_argument("--crop",required=True);p.add_argument("--image",required=True);a=p.parse_args();print(Predictor(a.checkpoint).predict(a.image,a.crop))
