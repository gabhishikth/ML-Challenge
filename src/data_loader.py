"""TSV loading helpers."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

def read_tsv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t",
        dtype={c: "string" for c in SOURCE_COLUMNS},
        keep_default_na=True, nrows=nrows,
    )

def iter_tsv_chunks(path: str | Path, chunksize: int = 50_000) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path, sep="\t",
        dtype={c: "string" for c in SOURCE_COLUMNS},
        keep_default_na=True, chunksize=chunksize,
    )

def load_sources(directory: str | Path, prefix: str, nrows: int | None = None):
    directory = Path(directory)
    return {
        f"source{i}": read_tsv(directory / f"{prefix}_source{i}.tsv", nrows)
        for i in (1, 2, 3)
    }

def load_reference_sources(directory: str | Path, prefix: str, nrows: int | None = None):
    sources = load_sources(directory, prefix, nrows)
    return pd.concat([
        sources["source2"].assign(record_source="source2"),
        sources["source3"].assign(record_source="source3"),
    ], ignore_index=True)

def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    frame = pd.read_csv(
        path, sep="\t",
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
        keep_default_na=False,
    )
    return {
        str(row.source1_entity_id): {
            x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()
        }
        for row in frame.itertuples(index=False)
    }
