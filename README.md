# Fake News Detection: RNN vs LSTM vs GRU

A project on classifying news as real or fake using recurrent neural networks.

## Description
A comparison of three recurrent neural network (RNN) architectures—RNN, LSTM, and GRU—using the FakeNewsNet dataset. The models predict whether a news item is fake (0) or real (1) based on its headline, URL, and source domain.

## Data
- Dataset: https://www.kaggle.com/datasets/algord/fake-news
- Features: `title`, `news_url`, and `source_domain` combined into a single text field
- Target variable: `real` (0 — fake, 1 — real)
- Split: balanced test set (1,000 fake + 1,000 real samples), with the remainder used for training

## Approach
1. Preprocessing: combining text fields, removing missing values, tokenization using spaCy, removing stop words, and limiting the vocabulary size.
2. Embeddings: Word2Vec (dimensionality 64), trained on the training corpus, with weights frozen (`freeze=True`).
3. Class balancing: `BCEWithLogitsLoss` with a `pos_weight` parameter to compensate for class imbalance.
4. Regularization: Dropout, early stopping (patience=3).
5. Evaluation: Accuracy, F1 Score, ROC AUC.

## Results
| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| RNN   | 0.7280   | 0.7258   | 0.8055  |
| LSTM  | 0.7760   | 0.7617   | 0.8520  |
| GRU   | **0.7805** | **0.7851** | **0.8537** |

**Conclusion:** Both LSTM and GRU significantly outperformed the standard RNN. GRU yielded the best results—it has fewer parameters, trains faster, and generalizes better on a relatively small dataset (20,866 examples). For short texts, the additional forget gate in the LSTM provides no advantage.
