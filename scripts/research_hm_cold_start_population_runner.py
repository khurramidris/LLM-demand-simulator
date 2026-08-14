from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

import research_hm_cold_start_population as exp


def _load_articles_from_locked_hf_revision(article_ids: set[int]):
    """Load only H&M article metadata from the exact locked HF dataset revision.

    This wrapper fixes repository-layout discovery only. It does not alter any
    target, split, baseline, metric, tuning grid, or preregistered decision gate.
    """
    api = HfApi()
    files = api.list_repo_files(
        repo_id=exp.ARTICLE_DATASET,
        repo_type="dataset",
        revision=exp.ARTICLE_DATASET_REVISION,
    )
    candidates = [
        f for f in files
        if "article" in f.lower()
        and (f.lower().endswith(".parquet") or f.lower().endswith(".csv"))
        and "image" not in f.lower()
    ]
    if not candidates:
        raise RuntimeError(
            f"No article metadata CSV/Parquet found at {exp.ARTICLE_DATASET}@{exp.ARTICLE_DATASET_REVISION}. "
            f"Repo files sample: {files[:50]}"
        )

    def priority(name: str):
        lower = name.lower()
        return (
            0 if lower.startswith("articles/") else 1,
            0 if "raw/articles" in lower else 1,
            0 if lower.endswith(".parquet") else 1,
            len(name),
            name,
        )

    filename = sorted(candidates, key=priority)[0]
    local = hf_hub_download(
        repo_id=exp.ARTICLE_DATASET,
        repo_type="dataset",
        revision=exp.ARTICLE_DATASET_REVISION,
        filename=filename,
    )
    path = Path(local)
    if filename.lower().endswith(".parquet"):
        art = pd.read_parquet(path)
    else:
        art = pd.read_csv(path)

    if "article_id" not in art.columns:
        raise RuntimeError(f"Chosen H&M article file lacks article_id: {filename}; columns={list(art.columns)}")
    art["article_id"] = pd.to_numeric(art["article_id"], errors="coerce").astype("Int64")
    art = art[art["article_id"].isin(article_ids)].copy()
    art["article_id"] = art["article_id"].astype("int64")
    art = art.drop_duplicates("article_id")

    fingerprint = hashlib.sha256(
        f"{exp.ARTICLE_DATASET}@{exp.ARTICLE_DATASET_REVISION}:{filename}".encode("utf-8")
    ).hexdigest()
    print(f"Locked H&M article metadata file: {filename}")
    return art, fingerprint


exp._load_articles = _load_articles_from_locked_hf_revision

if __name__ == "__main__":
    exp.main()
