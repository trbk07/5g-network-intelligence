"""Evaluate saved binary probabilities; never fit or select decision thresholds."""

import numpy as np
from sklearn.metrics import average_precision_score


def nested_probability_audit(warning, critical, mask):
    """Describe P(Y>=P95)>P(Y>=P90) violations without correcting forecasts."""
    warning,critical,mask=np.asarray(warning),np.asarray(critical),np.asarray(mask)
    if warning.shape!=critical.shape or mask.shape!=warning.shape or mask.dtype!=bool:
        raise ValueError("Aligned probability arrays and boolean observed mask required")
    a,b=warning[mask],critical[mask]
    if (not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or ((a<0)|(a>1)|(b<0)|(b>1)).any()):
        raise ValueError("Observed probabilities must be finite and in [0,1]")
    excess=np.maximum(0,b-a)
    violations=excess>1e-12
    return {"n_valid":len(a),"nesting_violations":int(violations.sum()),
        "nesting_violation_fraction":float(violations.mean()),"maximum_probability_excess":float(excess.max()),
        "tolerance":1e-12,"probabilities_corrected":False}


def probability_metrics(labels, probability):
    labels,probability=np.asarray(labels),np.asarray(probability,dtype=float)
    if labels.ndim!=1 or labels.shape!=probability.shape:
        raise ValueError("Aligned flat labels and probabilities required")
    observed=np.isfinite(labels)
    y,p=labels[observed],probability[observed]
    if (not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all()
            or ((p<0)|(p>1)).any()):
        raise ValueError("Observed binary labels and finite probabilities in [0,1] required")
    clipped=np.clip(p,1e-15,1-1e-15)
    predicted=p>=0.5
    positive=y==1
    tp=int((predicted&positive).sum())
    fp=int((predicted&~positive).sum())
    fn=int((~predicted&positive).sum())
    tn=int((~predicted&~positive).sum())
    return {"n_valid":len(y),"positive_events":int(positive.sum()),"base_rate":float(y.mean()),
        "brier":float(np.mean((p-y)**2)),
        "log_loss":float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
        "average_precision":float(average_precision_score(y,p)) if positive.any() else None,
        "TP":tp,"FP":fp,"FN":fn,"TN":tn,
        "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "flags":"|".join(flag for condition,flag in ((not positive.any(),"no_actual_positives"),
            (not predicted.any(),"no_predicted_positives")) if condition)}
