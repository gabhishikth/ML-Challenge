"""Entity-level F0.5 evaluation and threshold tuning."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def fbeta(p, r, beta=0.5):
    if p == 0 and r == 0: return 0.0
    b2 = beta * beta
    return (1+b2)*p*r/(b2*p+r)

def evaluate_predictions(predictions, ground_truth, source1_ids=None):
    ids = list(source1_ids if source1_ids is not None else ground_truth.keys())
    scores=[]; ps=[]; rs=[]; singleton_correct=0; predicted_count=0
    for eid in ids:
        pred=set(predictions.get(eid,set())); truth=set(ground_truth.get(eid,set()))
        predicted_count += len(pred)
        if not truth:
            ok = not pred
            singleton_correct += int(ok)
            scores.append(1.0 if ok else 0.0); ps.append(1.0 if ok else 0.0); rs.append(1.0 if ok else 0.0)
            continue
        tp=len(pred & truth)
        p=tp/len(pred) if pred else 0.0
        r=tp/len(truth)
        ps.append(p); rs.append(r); scores.append(fbeta(p,r))
    n=len(ids)
    return {
        "macro_f0_5": sum(scores)/n if n else 0.0,
        "macro_precision": sum(ps)/n if n else 0.0,
        "macro_recall": sum(rs)/n if n else 0.0,
        "singleton_accuracy": singleton_correct/n if n else 0.0,
        "predicted_matches": float(predicted_count),
        "entities": float(n),
    }

def predictions_at_threshold(scored_pairs, threshold):
    if scored_pairs.empty: return {}
    selected=scored_pairs[scored_pairs["match_probability"] >= threshold]
    if selected.empty: return {}
    return selected.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()

def threshold_sweep(scored_pairs, ground_truth, source1_ids, thresholds):
    rows=[]
    for t in thresholds:
        metrics=evaluate_predictions(predictions_at_threshold(scored_pairs,t),ground_truth,source1_ids)
        rows.append({"threshold":t,**metrics})
    return pd.DataFrame(rows)

def blocking_recall(candidate_pairs, ground_truth):
    grouped=candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
    truth=[(e,t) for e,t in ground_truth.items() if t]
    return sum(t.issubset(grouped.get(e,set())) for e,t in truth)/len(truth) if truth else 1.0

def save_experiment(metrics, directory="experiments"):
    d=Path(directory); d.mkdir(parents=True,exist_ok=True)
    ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p=d/f"experiment_{ts}.json"
    p.write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
    return p
