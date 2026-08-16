"""Example registry: specs and dataset loaders for the experiment harness.

Three domains: transformer NLP (ModernBERT pooled embeddings extracted by
embed_extract.py), classical NLP (TF-IDF words, character n-grams, averaged
GloVe vectors, engineered spam features), and beyond NLP (pixels, synthetic
selection benchmarks, molecular descriptors, sensor features).
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

CAP = 4000
SEED = 42


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    title: str
    domain: str  # "transformer NLP" | "classical NLP" | "beyond NLP"
    task: str
    blurb: str
    source: str


def _embedding_spec(name: str, title: str, blurb: str, source: str) -> ExampleSpec:
    return ExampleSpec(
        name=name, title=title, domain="transformer NLP", task="classification",
        blurb=blurb + " Features are mask-aware mean plus variance pooled "
        "ModernBERT-base hidden states (1,536 unnamed dimensions), ranked "
        "straight from the numpy matrix.",
        source=source,
    )


SPECS: dict[str, ExampleSpec] = {
    spec.name: spec
    for spec in [
        _embedding_spec(
            "modernbert_sst2", "SST-2 sentiment from ModernBERT embeddings",
            "Binary sentiment on movie-review sentences; the deep-dive "
            "methodology page for this data is "
            "[modernbert_sentiment.md](modernbert_sentiment.md).",
            "stanfordnlp/sst2, 4,000 sentences",
        ),
        _embedding_spec(
            "modernbert_agnews", "AG News topics from ModernBERT embeddings",
            "Four-way news topic classification.",
            "ag_news, 4,000 headlines with descriptions",
        ),
        _embedding_spec(
            "modernbert_emotion", "Emotion recognition from ModernBERT embeddings",
            "Six-way emotion classification of short messages.",
            "dair-ai/emotion, 4,000 messages",
        ),
        _embedding_spec(
            "modernbert_imdb", "IMDB reviews from ModernBERT embeddings",
            "Binary sentiment over long-form movie reviews (256-token window).",
            "imdb, 3,000 reviews",
        ),
        _embedding_spec(
            "modernbert_trec", "TREC question types from ModernBERT embeddings",
            "Six-way question-type classification of short questions.",
            "trec, 4,000 questions",
        ),
        _embedding_spec(
            "modernbert_offensive", "Offensive language from ModernBERT embeddings",
            "Binary offensive-language detection on tweets.",
            "tweet_eval/offensive, 4,000 tweets",
        ),
        _embedding_spec(
            "modernbert_subjectivity", "Subjectivity from ModernBERT embeddings",
            "Subjective vs objective sentence classification.",
            "SetFit/subj, 4,000 sentences",
        ),
        _embedding_spec(
            "modernbert_sms_spam", "SMS spam from ModernBERT embeddings",
            "Binary spam detection on SMS messages.",
            "sms_spam, 4,000 messages",
        ),
        ExampleSpec(
            "glove_sst2", "SST-2 sentiment from averaged GloVe vectors",
            "classical NLP", "classification",
            "The classical dense text representation: 300-dimensional GloVe "
            "word vectors averaged over each sentence, no context. A direct "
            "contrast with the contextual ModernBERT run on the same task.",
            "stanfordnlp/sst2 with glove-wiki-gigaword-300, 4,000 sentences",
        ),
        ExampleSpec(
            "tfidf_sst2", "SST-2 sentiment from TF-IDF words",
            "classical NLP", "classification",
            "Word-level TF-IDF (top 2,000 terms), so every ranked feature is "
            "a literal word and the consensus plot reads as a vocabulary of "
            "sentiment.",
            "stanfordnlp/sst2, 4,000 sentences",
        ),
        ExampleSpec(
            "tfidf_newsgroups", "20 Newsgroups from TF-IDF words",
            "classical NLP", "classification",
            "Four newsgroups (atheism, graphics, hockey, space) as word-level "
            "TF-IDF with headers, footers, and quotes stripped; ranked "
            "features are topic vocabulary.",
            "fetch_20newsgroups train subset, 4 categories",
        ),
        ExampleSpec(
            "chargram_langid", "Language identification from character n-grams",
            "classical NLP", "classification",
            "Six European languages from character 1-3 gram TF-IDF (top "
            "2,000 n-grams), the classical language-ID recipe.",
            "papluca/language-identification, 6 languages x 600 texts",
        ),
        ExampleSpec(
            "spambase_engineered", "Spambase engineered email features",
            "classical NLP", "classification",
            "Hand-engineered email features from 1999: word and character "
            "frequencies plus capital-run statistics. Classical feature "
            "engineering, fully named and interpretable.",
            "OpenML spambase (id 44), 4,601 emails, 57 features",
        ),
        ExampleSpec(
            "breast_cancer", "Breast cancer morphology",
            "beyond NLP", "classification",
            "The classic tabular case: 30 named tumor morphology features "
            "and a binary malignancy label.",
            "scikit-learn breast cancer dataset",
        ),
        ExampleSpec(
            "diabetes", "Diabetes progression",
            "beyond NLP", "regression",
            "Ten standardized clinical features against a continuous "
            "disease-progression target; the probe metric is R2.",
            "scikit-learn diabetes dataset",
        ),
        ExampleSpec(
            "digits_pixels", "Handwritten digits from raw pixels",
            "beyond NLP", "classification",
            "8x8 grayscale digits with each pixel as a named feature, so the "
            "ranking is a spatial saliency map.",
            "scikit-learn digits dataset, 1,797 images",
        ),
        ExampleSpec(
            "mnist_pixels", "MNIST from raw pixels",
            "beyond NLP", "classification",
            "28x28 MNIST pixels passed as an unnamed numpy matrix (784 "
            "generated IDs), ten classes.",
            "OpenML mnist_784 (id 554), 4,000-image subset",
        ),
        ExampleSpec(
            "madelon", "Madelon synthetic selection benchmark",
            "beyond NLP", "classification",
            "The NIPS 2003 feature-selection benchmark: 20 informative "
            "features hidden among 480 engineered probes and noise, built to "
            "defeat naive selectors.",
            "OpenML madelon (id 1485), 2,600 samples, 500 features",
        ),
        ExampleSpec(
            "bioresponse", "Molecular bioresponse descriptors",
            "beyond NLP", "classification",
            "Predicting a biological response from 1,776 molecular "
            "descriptors, a wide chemistry matrix.",
            "OpenML Bioresponse (id 4134), 3,751 molecules",
        ),
        ExampleSpec(
            "har_sensors", "Human activity from smartphone sensors",
            "beyond NLP", "classification",
            "Six activities from 561 engineered accelerometer and gyroscope "
            "features.",
            "OpenML har (id 1478), 4,000-sample subset",
        ),
    ]
}


def _cap(X, y, cap: int = CAP):
    if len(y) <= cap:
        return X, y
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(y), size=cap, replace=False))  # (cap,)
    X_capped = X.iloc[rows].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[rows]
    y_capped = np.asarray(y)[rows]
    return X_capped, y_capped


def _load_artifact(artifacts: Path, key: str):
    folder = artifacts / key
    E = np.load(folder / "embeddings.npy")  # (n, p)
    y = np.load(folder / "labels.npy")  # (n,)
    return E, y


def _tfidf(texts, labels, analyzer: str = "word", ngram_range=(1, 1)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=2000, min_df=3, analyzer=analyzer, ngram_range=ngram_range,
        stop_words="english" if analyzer == "word" else None,
    )
    M = vectorizer.fit_transform(texts).toarray().astype(np.float32)  # (n, p)
    X = pd.DataFrame(M, columns=vectorizer.get_feature_names_out())
    return X, np.asarray(labels)


def _openml(data_id: int):
    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = bundle.data.select_dtypes("number").astype(np.float32)
    y = bundle.target.to_numpy()
    return X, y


def load_dataset(name: str, artifacts: Path):
    if name.startswith("modernbert_") or name == "glove_sst2":
        return _load_artifact(artifacts, name)

    if name == "tfidf_sst2":
        from datasets import load_dataset as hf_load

        rows = hf_load("stanfordnlp/sst2", split="train").shuffle(seed=SEED).select(range(CAP))
        return _tfidf(rows["sentence"], rows["label"])
    if name == "tfidf_newsgroups":
        from sklearn.datasets import fetch_20newsgroups

        bundle = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes"),
            categories=["alt.atheism", "comp.graphics", "rec.sport.hockey", "sci.space"],
        )
        return _tfidf(bundle.data, bundle.target)
    if name == "chargram_langid":
        from datasets import load_dataset as hf_load

        rows = hf_load("papluca/language-identification", split="train")
        frame = rows.to_pandas()
        keep = frame[frame["labels"].isin(["en", "fr", "de", "es", "it", "pt"])]
        keep = keep.groupby("labels", group_keys=False).sample(n=600, random_state=SEED)
        return _tfidf(keep["text"].tolist(), keep["labels"].to_numpy(), analyzer="char", ngram_range=(1, 3))
    if name == "spambase_engineered":
        return _openml(44)
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        bundle = load_breast_cancer(as_frame=True)
        return bundle.data, bundle.target.to_numpy()
    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        bundle = load_diabetes(as_frame=True)
        return bundle.data, bundle.target.to_numpy()
    if name == "digits_pixels":
        from sklearn.datasets import load_digits

        bundle = load_digits(as_frame=True)
        return bundle.data, bundle.target.to_numpy()
    if name == "mnist_pixels":
        from sklearn.datasets import fetch_openml

        bundle = fetch_openml(data_id=554, as_frame=False, parser="auto")
        X, y = _cap(bundle.data.astype(np.float32), bundle.target)
        return X, y
    if name == "madelon":
        return _openml(1485)
    if name == "bioresponse":
        return _openml(4134)
    if name == "har_sensors":
        X, y = _openml(1478)
        return _cap(X, y)
    raise ValueError(f"Unknown example {name!r}.")
