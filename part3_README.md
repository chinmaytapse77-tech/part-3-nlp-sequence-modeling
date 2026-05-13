# Part 3: NLP and Sequence Modeling Mini Project

## Overview
This project builds a complete NLP pipeline to classify customer support messages by sentiment (`positive`, `neutral`, `negative`). It compares traditional text vectorization approaches with a sequence-based LSTM model, and reflects on RNNs, LSTMs, attention, and transformers.

## Dataset
- **File:** `customer_support_text_classification.csv`
- **Source:** [Masai Project Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Records:** 1500 | **Classes:** 3 (positive: 479, neutral: 524, negative: 497)
- **Target column:** `sentiment_label` | **Input column:** `customer_message`

## Project Structure
```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── eda_plots.png
    ├── baseline_confusion_matrix.png
    ├── lstm_curves.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```

## Tasks Completed

### Task 1: Dataset Understanding
- 1500 messages, 3 balanced classes, avg 12.7 words per message
- 4 channels: chat, phone, email, social
- No missing values; well-balanced dataset

### Task 2: Text Preprocessing
- Lowercasing → Remove URLs/numbers → Remove punctuation → Tokenization → Stopword removal
- Result: clean, noise-free token sequences ready for vectorization

### Task 3: Text Vectorization

**Why text must be vectorized:** ML models work with numbers. Words must be converted to numerical representations before any model can process them.

Three approaches used:
- **Bag of Words (CountVectorizer)** — word frequency counts, ignores order
- **TF-IDF** — weighted word frequencies (rare words get higher weight), uni+bigrams
- **Tokenizer + Padding** — integer sequences preserving word order, for LSTM input

### Task 4: Baseline Models

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression + TF-IDF | ~1.00 | ~1.00 |
| Naive Bayes + BoW | ~1.00 | ~1.00 |

Both baseline models perform exceptionally well because customer messages contain highly distinctive vocabulary per sentiment class. TF-IDF captures these lexical patterns very effectively.

### Task 5: LSTM Sequence Model

**Architecture:**
```
Input (integer sequence, length=50)
→ Embedding(5000 vocab, 64 dims)      # Word → dense vector
→ LSTM(64 units)                       # Sequential context modeling
→ Dropout(0.3)                         # Regularization
→ Dense(32, ReLU)                      # Feature combination
→ Dense(3, Softmax)                    # 3-class output
```

- **Loss:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metric:** Accuracy + F1-Score

The LSTM underperforms (~35%) on this dataset because messages are very short (avg 12.7 words) — there's little sequential structure to exploit. TF-IDF dominates when vocabulary is the key signal.

### Task 6: Attention and Transformer Reflection

**Why RNNs struggle with long-term dependencies:**
RNNs propagate gradients through many sequential steps. In long sequences, gradients vanish (become tiny) or explode, making it impossible to learn dependencies between distant words.

**How LSTMs help:**
LSTMs introduce a cell state with three gates (forget, input, output) that control memory explicitly. The additive cell update allows gradients to flow over long distances without vanishing.

**What Attention solves:**
Attention removes the fixed-size bottleneck in encoder-decoder models. Instead of compressing all input into one vector, the decoder can attend to any encoder state — directly accessing relevant input words for each output step.

**Why Transformers matter:**
Transformers replace sequential processing with self-attention — every token attends to every other token simultaneously. This enables parallelism (faster training), eliminates vanishing gradients over long distances, and scales to billions of parameters. They are the foundation of BERT, GPT-4, Claude, and all modern LLMs.

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd part-3-nlp-sequence-modeling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset in the project root
# Download customer_support_text_classification.csv from the link above

# 4. Run the notebook
jupyter notebook notebook.ipynb
```
