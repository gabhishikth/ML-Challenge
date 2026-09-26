#!/usr/bin/env bash
set -e
source .venv/bin/activate
python3 -m src.train --dataset-root dataset --sample-source1 1000 --model-path models/entity_resolution_model.joblib
python3 -m src.predict --dataset-root dataset --model-path models/entity_resolution_model.joblib --output-dir output/full_submission --chunk-size 5000
