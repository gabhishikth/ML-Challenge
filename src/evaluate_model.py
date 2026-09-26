"""Run blocking/model evaluation on a held-out Source-1 sample."""

from __future__ import annotations
import argparse
from pathlib import Path
from .data_loader import read_tsv, load_ground_truth
from .blocking import HybridBlocker, BlockingConfig, evaluate_blocking_recall
from .features import build_pair_features
from .train import load_model, predict_probabilities
from .evaluate import threshold_sweep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",default="dataset")
    ap.add_argument("--model-path",default="models/entity_resolution_model.joblib")
    ap.add_argument("--source1-rows",type=int,default=1000)
    args=ap.parse_args()
    root=Path(args.dataset_root); td=root/"train"
    s1=read_tsv(td/"train_source1.tsv",nrows=args.source1_rows)
    ref=__import__("pandas").concat([
        read_tsv(td/"train_source2.tsv").assign(record_source="source2"),
        read_tsv(td/"train_source3.tsv").assign(record_source="source3")
    ],ignore_index=True)
    gt=load_ground_truth(td/"train_ground_truth.tsv")
    blocker=HybridBlocker(BlockingConfig(max_candidates_per_entity=300,tfidf_top_k=150)).fit(ref)
    pairs,stats=blocker.generate(s1)
    stats["blocking_recall"]=evaluate_blocking_recall(pairs,gt)
    feats=build_pair_features(s1,ref,pairs)
    bundle=load_model(args.model_path)
    feats["match_probability"]=predict_probabilities(bundle,feats)
    sweep=threshold_sweep(feats,gt,s1.entity_id.astype(str),[x/100 for x in range(30,96,5)])
    print(sweep.sort_values("macro_f0_5",ascending=False).to_string(index=False))
    print("blocking:",stats)

if __name__=="__main__":
    main()
