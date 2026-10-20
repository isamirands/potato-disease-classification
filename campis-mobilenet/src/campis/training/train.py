from __future__ import annotations
from pathlib import Path
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from .sampler import HierarchicalCropSampler
from .checkpoint import save_checkpoint
def _device(name): return torch.device("cuda" if name=="auto" and torch.cuda.is_available() else "mps" if name=="auto" and getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu" if name=="auto" else name)
def _step(model, batch, optimizer=None, amp=False, device="cpu"):
    images,crops,labels,_=batch; images,labels=images.to(device),labels.to(device); total=torch.zeros((),device=device); n=0
    grouped=model(images,crops)
    for crop,(idx,logits) in grouped.items(): total += nn.functional.cross_entropy(logits, labels.index_select(0,idx))*len(idx); n+=len(idx)
    loss=total/max(n,1)
    if optimizer: optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    return float(loss.detach())
def train(model, dataset, crops, config, val_dataset=None):
    present = {(row.crop, row.disease) for _, row in dataset.frame.iterrows()}
    missing = [f"{crop}/{disease}" for crop, meta in crops.items() for disease in meta["classes"] if (crop, disease) not in present]
    if missing:
        raise ValueError("Entrenamiento bloqueado: faltan clases verificadas: " + ", ".join(missing))
    device=_device(config.get("device","auto")); model.to(device); out=Path(config.get("output_dir","artifacts/checkpoints")); out.mkdir(parents=True,exist_ok=True)
    workers = int(config.get("num_workers", 0))
    # Spawn-based Windows multiprocessing is fragile with third-party image
    # transforms.  A single process is deterministic and avoids silent worker
    # failures; it can be increased on Linux after profiling.
    if os.name == "nt": workers = 0
    sampler=HierarchicalCropSampler(dataset.frame,crops,seed=config.get("seed",42)); loader=DataLoader(dataset,batch_size=config["batch_size"],sampler=sampler,collate_fn=__import__("campis.datasets.loader",fromlist=["collate"]).collate,num_workers=workers)
    print(f"[CAMPIS] Dispositivo: {device} | imágenes de entrenamiento: {len(dataset)} | lotes/época: {len(loader)} | workers: {workers}", flush=True)
    stages=[("a",config["stage_a"],False), ("b",config["stage_b"],True)]; best=float("inf")
    for stage,opts,unfreeze in stages:
        for p in model.features.parameters(): p.requires_grad=unfreeze
        optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=opts.get("learning_rate",opts.get("head_learning_rate")),weight_decay=opts["weight_decay"])
        print(f"[CAMPIS] Etapa {stage.upper()} | {'fine-tuning del backbone' if unfreeze else 'backbone congelado'} | {opts['epochs']} épocas | lr={optimizer.param_groups[0]['lr']}", flush=True)
        for epoch in range(opts["epochs"]):
            model.train(); losses=[]; started=time.perf_counter(); total=len(loader); interval=max(1,int(config.get("log_interval",25)))
            for step, batch in enumerate(loader, start=1):
                losses.append(_step(model,batch,optimizer,device=device))
                if step % interval == 0 or step == total:
                    elapsed=time.perf_counter()-started; mean=sum(losses)/len(losses)
                    print(f"[CAMPIS] {stage.upper()} época {epoch+1}/{opts['epochs']} | lote {step}/{total} ({step/total:.0%}) | pérdida={mean:.4f} | {elapsed:.0f}s", flush=True)
            metric=sum(losses)/max(1,len(losses)); elapsed=time.perf_counter()-started
            print(f"[CAMPIS] Fin {stage.upper()} época {epoch+1}/{opts['epochs']} | pérdida media={metric:.4f} | duración={elapsed:.1f}s", flush=True)
            if metric<best:
                best=metric; save_checkpoint(out/"best_model.pt",model,optimizer,None,epoch,crops,config,{"train_loss":metric,"stage":stage})
                print(f"[CAMPIS] Nuevo mejor checkpoint guardado: {out/'best_model.pt'}", flush=True)
    return out/"best_model.pt"
