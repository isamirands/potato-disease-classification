import argparse,sys,yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from campis.datasets import load_crops,ManifestDataset
from campis.inference.preprocess import training_transform
from campis.models import MultiCropMobileNet
from campis.training.train import train
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/train.yaml"); parser.add_argument("--smoke-test",action="store_true")
    args=parser.parse_args(); config=yaml.safe_load(open(args.config,encoding="utf8")); crops=load_crops(config["crop_config"])
    if args.smoke_test:
        import torch
        model=MultiCropMobileNet(crops,pretrained=False); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        images=torch.rand(2,3,config["image_size"],config["image_size"]); logits=model(images,"tomato"); loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])); optimizer.zero_grad(); loss.backward(); optimizer.step(); print({"smoke_test":"ok","loss":float(loss.detach())})
    else:
        dataset=ManifestDataset(config["manifest_path"],crops,"train",training_transform(config["image_size"]))
        print(train(MultiCropMobileNet(crops),dataset,crops,config))

if __name__ == "__main__":
    main()
