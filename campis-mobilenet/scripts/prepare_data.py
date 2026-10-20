import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from campis.datasets.mappings import load_crops
from campis.datasets.prepare import build_manifest
from campis.datasets.download import write_download_instructions
p=argparse.ArgumentParser();p.add_argument("--config",default="configs/datasets.yaml");a=p.parse_args(); c=yaml.safe_load(open(a.config,encoding="utf8")); write_download_instructions(); print(build_manifest(c["raw_dir"],c["manifest_path"],load_crops("configs/crops.yaml"),{k:v.get("version","unknown") for k,v in c["sources"].items()},c["seed"]))
