# *Cyberbullying & Threat Detection using GloVe + TCN*

A deep learning NLP model that classifies harmful communication on social platforms using GloVe word embeddings and a Temporal Convolutional Network (TCN). Achieved 87% accuracy on real user comment data.

---

## *Project Overview*

This project detects online threats and cyberbullying in text comments. It combines traditional NLP techniques with deep learning to build an automated system that flags toxic or suspicious language.

---

## *📁 Dataset*

- Source: "Suspicious Communication on Social Platforms"
- Fields:
  - `comments`: Raw text data (user posts)
  - `tagging`: Labels (harmful or not)
- Goal: Binary classification (0 = safe, 1 = suspicious)

---

## * Preprocessing Pipeline*

1. Convert text to lowercase
2. Remove URLs, usernames, and special characters
3. Tokenize into words
4. Remove stopwords (like “the”, “is”, etc.)

---

## * Tokenization*

Tokenization is the process of splitting text into individual words or “tokens”.  
Each unique word is assigned an integer index using a `Tokenizer`.

```python
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
```

This prepares the data for the model by turning raw sentences into sequences of numbers.

---

## * Padding*

Since sentences are different lengths, we **pad** them to the same length using:

```python
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
```

This makes the input shape consistent for the neural network.

---

## * What is GloVe?*

**GloVe (Global Vectors)** is a pretrained word embedding model.  
It turns words into vectors based on **global co-occurrence statistics** (how often words appear together in large corpora like Wikipedia or Common Crawl).

### Example:
Words like **“hate”** and **“disgust”** often appear in similar contexts (e.g., “I hate this movie”, “I feel disgusted”), so GloVe places them close together in vector space.

```python
embedding_index['hate'] → [0.52, -0.14, ..., 0.37]  # 100-d vector
```

---

## * Embedding Matrix*

We create an **embedding matrix** to map each word in our vocabulary (from the tokenizer) to its GloVe vector:

```python
embedding_matrix[i] = embedding_index.get(word)
```

This matrix is used to initialize the neural network’s Embedding layer with **meaningful, pretrained word vectors**, instead of random values.

---

## *GloVe vs Word2Vec*

| Feature       | GloVe                           | Word2Vec                          |
|---------------|----------------------------------|-----------------------------------|
| Training Data | Global co-occurrence            | Local context (sliding window)   |
| Vectors       | Pretrained (static)             | Usually trained on your data     |
| Strength      | Fast, consistent embeddings     | Can adapt to custom domains       |

Both output vector representations of words, but GloVe is more global and pretrained.

---

## *TCN (Temporal Convolutional Network)*

A **TCN** is a neural network that handles sequence data like text or time series using **1D convolutions**. Unlike RNNs or LSTMs, it doesn’t process data step-by-step, but in parallel.

### Why TCN over LSTM?
- Faster training (parallelizable)
- No vanishing gradient issue
- Maintains long-range memory using *dilated convolutions*

```python
TCN(nb_filters=64, return_sequences=False)
```

This layer captures temporal dependencies in the sequence of word embeddings.

---

## *Model Architecture*

```python
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim,
              weights=[embedding_matrix], input_length=max_len, trainable=False),
    TCN(nb_filters=64, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
```

- **Embedding Layer**: Initialized with GloVe vectors
- **TCN Layer**: Learns sequential patterns in text
- **Dense + Dropout**: Classification layers
- **Sigmoid Output**: For binary classification

---

## *⚙Training Details*

- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Epochs**: 6
- **Batch Size**: 64

---

## *Evaluation*

- Achieved **87% accuracy** on test set
- Generated classification report (Precision, Recall, F1)
- Plotted training & validation curves

---

## *Final Takeaways*

- Tokenization prepares text for neural nets
- GloVe provides meaning-rich vectors for each word
- Embedding matrix links words to pretrained embeddings
- TCN is an efficient alternative to LSTMs for sequence learning
- Combined, they enable powerful NLP models for real-world applications

