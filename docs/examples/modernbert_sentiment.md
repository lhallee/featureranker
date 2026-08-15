# ModernBERT sentiment: ranking unnamed embedding dimensions

Nothing in featureranker requires named columns or a table. Here the
features are the 1,536 pooled hidden-state dimensions of
[answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) over
4,000 SST-2 sentences (mask-aware mean pooling concatenated
with mask-aware variance pooling, 768 dimensions each), and the target is
the sentiment label. Passing the bare numpy matrix assigns each dimension a
stable ID: f0000-f0767 are mean-pooled, f0768-f1535
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

The top 20 of 1,536 dimensions hold
41% of the total vote score, and the L1 path never activated
52 dimensions at all (92 parallel
liblinear fits located every entry point). Of the top 20,
19 are mean-pooled and 1 are variance-pooled dimensions.

| feature | score |
|---|---|
| f0211 | 4.5 |
| f0390 | 1.738 |
| f0569 | 1.71 |
| f0570 | 1.596 |
| f0092 | 1.043 |
| f0686 | 0.7097 |
| f0354 | 0.5579 |
| f0533 | 0.5549 |
| f0108 | 0.5472 |
| f0673 | 0.4218 |

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
