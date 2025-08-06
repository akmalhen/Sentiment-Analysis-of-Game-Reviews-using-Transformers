# 🧠 NLP Project: Transformer Models Benchmark for Text Classification

This project aims to compare the performance of several popular Transformer models for text classification tasks, using pre-prepared datasets. The models used include TinyBERT, BERT, RoBERTa, and DeBERTa.

---

## 📦 Installation

Install all the following dependencies before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install transformers datasets accelerate evaluate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ It is recommended to use a GPU (Google Colab or local with CUDA) to accelerate the transformer model training process.

---

## 📁 Project Structure

```
NLP_Project/
│
├── NLP_Project_9.ipynb # Main project notebook
├── dataset/ # Dataset folder (optional, depending on your dataset path)
└── README.md # Project documentation
```

---

## 🔍 Project Flow

### 1. Install & Import Libraries

Install and import libraries such as `transformers`, `torch`, and `sklearn`.

### 2. Check GPU

Verify that the GPU is available to run the transformer-based model.

### 3. Initialize Functions

Custom functions are prepared for the preprocessing, evaluation, and training pipeline.

### 4. Load Dataset

Read a CSV dataset containing text and classification labels.

### 5. Exploratory Data Analysis (EDA)

Exploring the data:
- Label distribution
- Text length
- Text statistics

### 6. Preprocessing

- Text cleaning
- Tokenization
- Label encoding
- Train/test data split

### 7. Modeling

Transformer models used:
- **TinyBERT**
- **BERT-base**
- **RoBERTa-base**
- **DeBERTa-base**

All models were trained using the `Trainer` from HuggingFace Transformers.

### 8. Evaluation

Evaluation is based on:
- Accuracy
- Precision
- Recall
- F1-score

---

## 🛠 Technology & Tools

- Python
- Jupyter Notebook
- HuggingFace Transformers
- PyTorch
- scikit-learn
- Matplotlib & Seaborn

---

## 📌 Notes

- Ensure the dataset is available and its path matches the one in the notebook.
- Can be run optimally on Google Colab (GPU).
- Suitable for initial text classification benchmarks using various transformer models.

---

## 👤 Development Team

- Akmal Hendrian Malik
- Alfito Faiz Rizqi
- Ignatius Kevin Wijaya

---

> Thank you for visiting this repo 🙏 Please fork, clone, and explore!
