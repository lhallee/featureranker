"""Extract pooled text representations for the embedding-based examples.

Needs torch + transformers + datasets (ModernBERT pooling) or gensim
(averaged GloVe); the ranking side never imports any of them, so the two
phases can run in different interpreters.

    python examples/embed_extract.py --dataset modernbert_agnews --out artifacts
    python examples/embed_extract.py --all --out artifacts
"""

import argparse
import json

import numpy as np

from pathlib import Path

from modernbert_sentiment import BATCH_SIZE, MODEL_NAME

SEED = 42

# name -> (hf path, hf config, split, text field, label field, cap, max_length)
TEXT_DATASETS = {
    "modernbert_sst2": ("stanfordnlp/sst2", None, "train", "sentence", "label", 4000, 128),
    "modernbert_agnews": ("ag_news", None, "train", "text", "label", 4000, 128),
    "modernbert_emotion": ("dair-ai/emotion", None, "train", "text", "label", 4000, 128),
    "modernbert_imdb": ("imdb", None, "train", "text", "label", 3000, 256),
    "modernbert_trec": ("trec", None, "train", "text", "coarse_label", 4000, 64),
    "modernbert_offensive": ("tweet_eval", "offensive", "train", "text", "label", 4000, 128),
    "modernbert_subjectivity": ("SetFit/subj", None, "train", "text", "label", 4000, 128),
    "modernbert_sms_spam": ("sms_spam", None, "train", "sms", "label", 4000, 128),
}


def _load_texts(name: str) -> tuple[list[str], np.ndarray, int]:
    from datasets import load_dataset as hf_load

    path, config, split, text_field, label_field, cap, max_length = TEXT_DATASETS[name]
    rows = hf_load(path, config, split=split).shuffle(seed=SEED)
    rows = rows.select(range(min(cap, len(rows))))
    labels = np.asarray(rows[label_field], dtype=np.int64)  # (n,)
    return list(rows[text_field]), labels, max_length


def extract_modernbert(name: str, out_root: Path) -> None:
    import torch

    from transformers import AutoModel, AutoTokenizer

    texts, labels, max_length = _load_texts(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()

    features = []
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + BATCH_SIZE], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            H = model(**encoded).last_hidden_state  # (b, l, d)
            mask = encoded["attention_mask"].unsqueeze(-1).to(H.dtype)  # (b, l, 1)
            counts = mask.sum(dim=1).clamp(min=1.0)  # (b, 1)
            mean = (H * mask).sum(dim=1) / counts  # (b, d)
            variance = ((H - mean.unsqueeze(1)) * mask).pow(2).sum(dim=1) / counts  # (b, d)
            features.append(torch.cat([mean, variance], dim=-1).float().cpu().numpy())

    _save(out_root / name, np.concatenate(features, axis=0), labels,
          {"model": MODEL_NAME, "max_length": max_length})


def extract_glove_sst2(out_root: Path) -> None:
    import re

    import gensim.downloader

    from datasets import load_dataset as hf_load

    rows = hf_load("stanfordnlp/sst2", split="train").shuffle(seed=SEED).select(range(4000))
    labels = np.asarray(rows["label"], dtype=np.int64)  # (n,)
    vectors = gensim.downloader.load("glove-wiki-gigaword-300")

    E = np.zeros((len(labels), 300), dtype=np.float32)  # (n, 300)
    for i, sentence in enumerate(rows["sentence"]):
        tokens = re.findall(r"[a-z']+", sentence.lower())
        hits = [vectors[token] for token in tokens if token in vectors]
        if hits:
            E[i] = np.mean(hits, axis=0)
    _save(out_root / "glove_sst2", E, labels, {"model": "glove-wiki-gigaword-300"})


def _save(folder: Path, E: np.ndarray, labels: np.ndarray, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", E)
    np.save(folder / "labels.npy", labels)
    meta.update({"n_samples": int(E.shape[0]), "n_features": int(E.shape[1])})
    (folder / "meta.json").write_text(json.dumps(meta))
    print(f"{folder.name}: {E.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*TEXT_DATASETS, "glove_sst2"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    chosen = [*TEXT_DATASETS, "glove_sst2"] if args.all else [args.dataset]
    for name in chosen:
        try:
            if name == "glove_sst2":
                extract_glove_sst2(args.out)
            else:
                extract_modernbert(name, args.out)
        except Exception as error:
            # keep extracting the remaining sets when one download breaks
            print(f"FAILED {name}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
