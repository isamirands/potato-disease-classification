import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from campis.export.export_onnx import export_onnx
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);a=p.parse_args();print(export_onnx(a.checkpoint))
