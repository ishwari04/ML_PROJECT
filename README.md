# 🧠 AI-Driven Sentiment Analysis on Company Reviews

## 📌 Overview
This project applies **Natural Language Processing (NLP)** to perform **sentiment analysis** on company reviews.  
It uses both **Machine Learning (Logistic Regression)** and **Deep Learning (USE + Dense Neural Network)** approaches to classify text reviews as *positive*, *neutral*, or *negative* — helping organizations better interpret employee and customer sentiments.

---

## 📂 Dataset
**Source:** Company review dataset containing textual reviews with numerical sentiment ratings (1–5).  
**Files Used:**
- `train.csv` — labeled data for training  
- `test.csv` — unlabeled reviews for prediction  
- `sample_submission.csv` — format for submission  

Each record includes:
- **Review:** textual feedback  
- **Rating:** numeric sentiment score  

---

## ⚙️ Preprocessing Pipeline
1. **Cleaning:** removed nulls, symbols, and stopwords  
2. **Normalization:** lowercasing and tokenization  
3. **Exploratory Data Analysis (EDA):**
   - Distribution of ratings  
   - Review length variation  
   - Token count analysis  
4. **Feature Extraction:**
   - TF-IDF vectorization (for ML model)
   - Universal Sentence Encoder embeddings (for DL model)

---

## 🧩 Models Implemented

### 🔹 Model 1 — Logistic Regression (Baseline ML)
- **Approach:** TF-IDF + Logistic Regression  
- **Purpose:** Establish a benchmark using classical ML  
- **Performance:**
  - Accuracy: ~75%  
  - F1-Score: ~0.72  

### 🔹 Model 2 — USE + Dense Neural Network (Deep Learning)
- **Approach:** USE embeddings + 3-layer Dense NN  
- **Architecture:**  
  Input(512) → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Output(Softmax)  
- **Performance:**
  - Accuracy: ~88%  
  - F1-Score: ~0.86  

---

## 📊 Comparative Performance

| Model | Type | Feature | Accuracy | F1-Score |
|-------|------|----------|-----------|-----------|
| Logistic Regression | Classical ML | TF-IDF | ~0.75 | ~0.72 |
| USE + Dense NN | Deep Learning | Sentence Embeddings | ~0.88 | ~0.86 |

> 🧠 Deep Learning model achieves higher contextual understanding compared to frequency-based TF-IDF features.

---

## 📈 Visualizations
- 📊 Rating Distribution Plot  
- 📦 Review Length & Token Boxplots  
- 🔲 Confusion Matrix for each model  
- 📉 Accuracy & Loss Curves (DL model)  
- 📚 Comparative Bar Graph: Accuracy vs F1-Score  

---

## 💡 Key Insights
- Logistic Regression performs well with frequency-based TF-IDF features.  
- Universal Sentence Encoder significantly improves contextual understanding.  
- Deep Neural Networks generalize better on unseen reviews.

---

## 🛠 Tech Stack
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`, `tensorflow_hub`  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 🚀 Future Work
- Integrate **BERT / DistilBERT** for transformer-based modeling  
- Add **Streamlit dashboard** for live sentiment visualization  
- Explore **aspect-based sentiment analysis**

---

## 👩‍💻 Author
**Ishwari Kakade**  
Lead Developer & Researcher — *AI-Driven Sentiment Analysis on Company Reviews*  
📧 [Your Email Here]

---

⭐ *If you find this project useful, feel free to star the repo and contribute!*
