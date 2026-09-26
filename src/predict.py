"""Chunked test prediction and submission generation."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .data_loader import read_tsv, iter_tsv_chunks, load_reference_sources
from .blocking import HybridBlocker, BlockingConfig
from .features import build_pair_features
from .train import load_model, predict_probabilities

def predict_chunked(dataset_root, model_path, output_dir, threshold=None, chunk_size=5000):
    root=Path(dataset_root); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    bundle=load_model(model_path)
    if threshold is None: threshold=float(bundle.get("best_threshold",0.80))

    ref=pd.concat([
        pd.read_csv(root/"test/test_source2.tsv",sep="\t",dtype="string").assign(record_source="source2"),
        pd.read_csv(root/"test/test_source3.tsv",sep="\t",dtype="string").assign(record_source="source3")
    ],ignore_index=True)

    blocker=HybridBlocker(BlockingConfig(max_candidates_per_entity=300,tfidf_top_k=150)).fit(ref)
    candidate_path=out/"candidate_pairs.tsv"
    matching_path=out/"matching_results.tsv"
    first=True

    for s1 in iter_tsv_chunks(root/"test/test_source1.tsv",chunksize=chunk_size):
        pairs,stats=blocker.generate(s1)
        feats=build_pair_features(s1,ref,pairs)
        if not feats.empty:
            feats["match_probability"]=predict_probabilities(bundle,feats)
        cmap=pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict() if not pairs.empty else {}
        matches={}
        if not feats.empty:
            selected=feats[feats.match_probability>=threshold]
            if not selected.empty:
                matches=selected.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()

        candidate_frame=pd.DataFrame({
            "source1_entity_id":s1.entity_id.astype(str),
            "candidate_entity_ids":[",".join(sorted(cmap.get(str(e),set()))) for e in s1.entity_id]
        })
        matching_frame=pd.DataFrame({
            "source1_entity_id":s1.entity_id.astype(str),
            "matched_entity_ids":[",".join(sorted(matches.get(str(e),set()))) for e in s1.entity_id]
        })
        candidate_frame.to_csv(candidate_path,sep="\t",index=False,mode="w" if first else "a",header=first)
        matching_frame.to_csv(matching_path,sep="\t",index=False,mode="w" if first else "a",header=first)
        first=False
        print({"processed_source1":len(s1),"candidate_rows":len(pairs),"blocking":stats,"threshold":threshold},flush=True)

    print(f"candidate_pairs: {candidate_path}")
    print(f"matching_results: {matching_path}")
    print(f"threshold: {threshold}")
    return candidate_path,matching_path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",default="dataset")
    ap.add_argument("--model-path",default="models/entity_resolution_model.joblib")
    ap.add_argument("--output-dir",default="output/full_submission")
    ap.add_argument("--threshold",type=float,default=None)
    ap.add_argument("--chunk-size",type=int,default=5000)
    args=ap.parse_args()
    predict_chunked(args.dataset_root,args.model_path,args.output_dir,args.threshold,args.chunk_size)

if __name__=="__main__":
    main()
