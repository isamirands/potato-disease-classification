from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from .metrics import crop_metrics
def evaluate(model, loader, crops, output_dir="artifacts/reports", device="cpu"):
    model.eval(); truth=defaultdict(list); pred=defaultdict(list)
    with torch.inference_mode():
        for images,names,labels,_ in loader:
            images=images.to(device)
            for crop,(idx,logits) in model(images,names).items():
                truth[crop].extend(labels.index_select(0,idx.cpu()).tolist()); pred[crop].extend(logits.argmax(1).cpu().tolist())
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    report={}; rows=[]
    for crop in crops:
        if not truth[crop]: continue
        labels=list(range(len(crops[crop]["classes"]))); names=crops[crop]["classes"]
        metrics=crop_metrics(truth[crop],pred[crop]); metrics["samples"]=len(truth[crop]); report[crop]=metrics
        matrix=confusion_matrix(truth[crop],pred[crop],labels=labels)
        fig,axis=plt.subplots(figsize=(max(6,len(names)*1.35),max(5,len(names)*1.15)))
        ConfusionMatrixDisplay(matrix,display_labels=names).plot(ax=axis,cmap="Blues",colorbar=False,xticks_rotation=35,values_format="d")
        axis.set_title(f"CAMPIS — matriz de confusión: {crops[crop]['display_name']}"); fig.tight_layout(); fig.savefig(out/f"confusion_{crop}.png",dpi=180); plt.close(fig)
        class_report=classification_report(truth[crop],pred[crop],labels=labels,target_names=names,output_dict=True,zero_division=0)
        for disease in names:
            item=class_report[disease]; rows.append({"crop":crop,"disease":disease,"support":int(item["support"]),"precision":item["precision"],"recall":item["recall"],"f1":item["f1-score"]})
        (out/f"confusion_{crop}.json").write_text(json.dumps({"labels":names,"matrix":matrix.tolist()},indent=2),encoding="utf-8")
    report["macro_f1_mean_across_crops"]=sum(x["f1_macro"] for crop,x in report.items() if crop in crops)/max(1,len([crop for crop in report if crop in crops]))
    (out/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); pd.DataFrame(rows).to_csv(out/"per_crop_metrics.csv",index=False); return report
