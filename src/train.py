"""Train the final hybrid entity-resolution matcher."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier

from .data_loader import read_tsv, iter_tsv_chunks, load_ground_truth
from .blocking import HybridBlocker, BlockingConfig, evaluate_blocking_recall
from .features import build_pair_features, FEATURE_COLUMNS
from .evaluate import threshold_sweep, save_experiment

def make_training_frame(features, ground_truth):
    x=features.copy()
    x["target"]=[float(c in ground_truth.get(s,set())) for s,c in zip(x.source1_entity_id,x.candidate_entity_id)]
    return x

def split_by_source1(df, fraction=0.2, random_state=42):
    if df.empty: return df.copy(),df.copy()
    g=GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=random_state)
    a,b=next(g.split(df,groups=df.source1_entity_id))
    return df.iloc[a].copy(),df.iloc[b].copy()

def fit_model(train, random_state=42):
    if train.empty or train.target.nunique()<2:
        raise ValueError("Training data must contain both positive and negative examples.")
    neg=(train.target==0).sum(); pos=(train.target==1).sum()
    # HistGradientBoosting is the dependency-light fallback. LightGBM is preferred when installed.
    try:
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(
            objective="binary", n_estimators=450, learning_rate=0.04,
            num_leaves=31, max_depth=7, min_child_samples=20,
            subsample=0.85, colsample_bytree=0.9,
            scale_pos_weight=min(neg/max(pos,1),10.0),
            random_state=random_state, n_jobs=-1,
        )
        backend="lightgbm"
    except ImportError:
        model=HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            max_depth=7, random_state=random_state,
        )
        backend="sklearn_hist_gradient_boosting"
    model.fit(train[FEATURE_COLUMNS],train.target)
    return {"model":model,"feature_columns":FEATURE_COLUMNS,"backend":backend,"best_threshold":0.80}

def load_model(path): return joblib.load(path)
def save_model(bundle,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,p)

def predict_probabilities(bundle, features):
    if features.empty: return pd.Series(dtype=float,index=features.index)
    return pd.Series(bundle["model"].predict_proba(features[bundle["feature_columns"]])[:,1],index=features.index)

def sample_reference(train_dir, positive_ids, negative_per_source=10000):
    frames=[]
    for n in (2,3):
        path=Path(train_dir)/f"train_source{n}.tsv"
        positives=[]
        negatives=[]
        remaining=negative_per_source
        for chunk in iter_tsv_chunks(path,chunksize=50000):
            p=chunk[chunk.entity_id.isin(positive_ids)]
            if not p.empty: positives.append(p.assign(record_source=f"source{n}"))
            if remaining>0:
                q=chunk[~chunk.entity_id.isin(positive_ids)].sample(
                    n=min(remaining,len(chunk)), random_state=42
                ) if len(chunk) else chunk
                if not q.empty:
                    negatives.append(q.assign(record_source=f"source{n}"))
                    remaining-=len(q)
            if remaining<=0 and positives and len(pd.concat(positives))>=len({x for x in positive_ids if x.startswith(f"S{n}-")}):
                break
        frames += positives + negatives
    if not frames: raise ValueError("Reference sample is empty.")
    return pd.concat(frames,ignore_index=True).drop_duplicates("entity_id")

def select_training_negatives(features, negative_per_entity=12):
    # Keep all positives, plus hard negatives (highest blocking similarity) and random negatives.
    pos=features[features.target==1]
    neg=features[features.target==0]
    hard=neg.sort_values(["source1_entity_id","blocking_similarity"],ascending=[True,False]).groupby("source1_entity_id").head(8)
    remaining=neg.drop(hard.index,errors="ignore")
    rnd=remaining.groupby("source1_entity_id",group_keys=False).apply(
        lambda g:g.sample(n=min(4,len(g)),random_state=42) if len(g) else g,
        include_groups=False
    ) if not remaining.empty else remaining
    if isinstance(rnd,pd.DataFrame) and not rnd.empty:
        # Restore ID columns if pandas version drops group labels.
        if "source1_entity_id" not in rnd.columns:
            rnd=remaining.loc[rnd.index]
    return pd.concat([pos,hard,rnd],ignore_index=True).drop_duplicates(["source1_entity_id","candidate_entity_id"])

def train_sample(dataset_root="dataset", sample_source1=1000, model_path="models/entity_resolution_model.joblib"):
    root=Path(dataset_root); td=root/"train"
    s1=read_tsv(td/"train_source1.tsv",nrows=sample_source1)
    gt=load_ground_truth(td/"train_ground_truth.tsv")
    truth={str(e):gt.get(str(e),set()) for e in s1.entity_id}
    pos_ids=set().union(*truth.values()) if truth else set()
    ref=sample_reference(td,pos_ids)
    blocker=HybridBlocker(BlockingConfig(max_candidates_per_entity=300,tfidf_top_k=150))
    pairs,stats=blocker.fit(ref).generate(s1)
    stats["blocking_recall"]=evaluate_blocking_recall(pairs,truth)
    feats=build_pair_features(s1,ref,pairs)
    train=make_training_frame(feats,truth)
    train,valid=split_by_source1(train)
    train=select_training_negatives(train)
    bundle=fit_model(train)
    valid_scored=valid.copy()
    valid_scored["match_probability"]=predict_probabilities(bundle,valid)
    sweep=threshold_sweep(valid_scored,gt,valid.source1_entity_id.unique(),np.arange(0.30,0.96,0.025))
    best=sweep.sort_values("macro_f0_5",ascending=False).iloc[0]
    bundle["best_threshold"]=float(best.threshold)
    save_model(bundle,model_path)
    exp={"mode":"sample","source1_rows":len(s1),"reference_rows":len(ref),"candidate_rows":len(pairs),
         "blocking":stats,"training_rows":len(train),"positive_pairs":int(train.target.sum()),
         "backend":bundle["backend"],"best_threshold":float(best.threshold),"validation":best.to_dict()}
    save_experiment(exp); print(json.dumps(exp,indent=2,default=str))
    return bundle

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",default="dataset")
    ap.add_argument("--sample-source1",type=int,default=1000)
    ap.add_argument("--model-path",default="models/entity_resolution_model.joblib")
    args=ap.parse_args()
    train_sample(args.dataset_root,args.sample_source1,args.model_path)

if __name__=="__main__":
    import json
    main()
