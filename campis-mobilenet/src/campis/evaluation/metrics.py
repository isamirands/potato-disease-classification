from __future__ import annotations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def crop_metrics(y_true,y_pred):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    return {"accuracy":accuracy_score(y_true,y_pred),"precision_macro":p,"recall_macro":r,"f1_macro":f}
