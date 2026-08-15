"""Rank pooled ModernBERT dimensions for sentiment; regenerates its docs page.

Two phases so the GPU stack and the ranking stack can live in different
interpreters (for example a system Python with CUDA torch, and a newer venv
with featureranker):

    python examples/modernbert_sentiment.py extract --out artifacts
    python examples/modernbert_sentiment.py rank --artifacts artifacts

extract needs torch + transformers + datasets and a GPU helps; rank needs
featureranker and writes docs/examples/modernbert_sentiment.md plus images.
Each text is embedded as mask-aware mean pooling concatenated with
mask-aware variance pooling of the final hidden states, so a base model with
hidden width 768 yields 1,536 unnamed features per input.
"""

import argparse
import json

import numpy as np

from pathlib import Path

MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET = "stanfordnlp/sst2"
N_SAMPLES = 4000
MAX_LENGTH = 128
BATCH_SIZE = 64


def extract(out_dir: Path) -> None:
    import torch

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()

    rows = load_dataset(DATASET, split=f"train[:{N_SAMPLES}]")
    texts = rows["sentence"]
    labels = np.asarray(rows["label"], dtype=np.int64)  # (n,)

    features = []
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            H = model(**encoded).last_hidden_state  # (b, l, d)
            mask = encoded["attention_mask"].unsqueeze(-1).to(H.dtype)  # (b, l, 1)
            counts = mask.sum(dim=1).clamp(min=1.0)  # (b, 1)
            mean = (H * mask).sum(dim=1) / counts  # (b, d)
            centered = (H - mean.unsqueeze(1)) * mask  # (b, l, d)
            variance = centered.pow(2).sum(dim=1) / counts  # (b, d)
            pooled = torch.cat([mean, variance], dim=-1)  # (b, 2d)
            features.append(pooled.float().cpu().numpy())

    E = np.concatenate(features, axis=0)  # (n, 2d)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", E)
    np.save(out_dir / "labels.npy", labels)
    (out_dir / "meta.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "dataset": DATASET,
        "n_samples": int(E.shape[0]),
        "n_features": int(E.shape[1]),
        "hidden_width": int(E.shape[1] // 2),
        "max_length": MAX_LENGTH,
        "device": device,
    }))
    print(f"extracted {E.shape} features on {device} -> {out_dir}")


def rank(artifacts: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from _pages import image_path, md_table, save_page
    from featureranker import (
        feature_ranking,
        plot_after_vote,
        plot_rank_heatmap,
        plot_rankings,
        voting,
    )

    E = np.load(artifacts / "embeddings.npy")  # (n, 2d)
    y = np.load(artifacts / "labels.npy")  # (n,)
    meta = json.loads((artifacts / "meta.json").read_text())
    d = meta["hidden_width"]

    result = feature_ranking(E, y, task="classification")
    vote_table = voting(result)

    plot_rankings(
        result, top_n=20, show=False, save=True,
        save_path=image_path("modernbert_sentiment_rankings.png"),
    )
    plot_rank_heatmap(
        result, top_n=20, show=False, save=True,
        save_path=image_path("modernbert_sentiment_heatmap.png"),
    )
    plot_after_vote(
        vote_table, top_n=20, show=False, save=True,
        save_path=image_path("modernbert_sentiment_vote.png"),
    )

    top20 = vote_table.head(20)
    top_share = top20["score"].sum() / vote_table["score"].sum()
    top_indices = [int(name[1:]) for name in top20["feature"]]
    n_mean = sum(1 for i in top_indices if i < d)
    l1 = result.diagnostics["l1"]

    page = f"""# ModernBERT sentiment: ranking unnamed embedding dimensions

Nothing in featureranker requires named columns or a table. Here the
features are the {meta['n_features']:,} pooled hidden-state dimensions of
[{meta['model']}](https://huggingface.co/{meta['model']}) over
{meta['n_samples']:,} SST-2 sentences (mask-aware mean pooling concatenated
with mask-aware variance pooling, {d} dimensions each), and the target is
the sentiment label. Passing the bare numpy matrix assigns each dimension a
stable ID: f0000-f{d - 1:04d} are mean-pooled, f{d:04d}-f{meta['n_features'] - 1:04d}
are variance-pooled.

```python
import numpy as np
from featureranker import feature_ranking, voting

E = np.load("embeddings.npy")   # (4000, 1536) pooled ModernBERT features
y = np.load("labels.npy")       # (4000,) sentiment labels
result = feature_ranking(E, y, task="classification")
vote_table = voting(result)
```

## A few dimensions carry the signal

The top 20 of {meta['n_features']:,} dimensions hold
{top_share:.0%} of the total vote score, and the L1 path never activated
{l1['n_never_entered']:,} dimensions at all ({l1['n_fits']} parallel
{l1['solver']} fits located every entry point). Of the top 20,
{n_mean} are mean-pooled and {20 - n_mean} are variance-pooled dimensions.

{md_table(vote_table, 10)}

![Aggregated dimension importance](../images/modernbert_sentiment_vote.png)

![Dimension ranks by method](../images/modernbert_sentiment_rankings.png)

![Rank heatmap](../images/modernbert_sentiment_heatmap.png)

## Reproduce

```bash
python examples/modernbert_sentiment.py extract --out artifacts
python examples/modernbert_sentiment.py rank --artifacts artifacts
```

The two phases can run in different environments: extract needs torch,
transformers, and datasets; rank needs featureranker.
"""
    print("wrote", save_page("modernbert_sentiment", page))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    extract_cmd = subcommands.add_parser("extract")
    extract_cmd.add_argument("--out", type=Path, default=Path("artifacts"))
    rank_cmd = subcommands.add_parser("rank")
    rank_cmd.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if args.command == "extract":
        extract(args.out)
    else:
        rank(args.artifacts)


if __name__ == "__main__":
    main()
