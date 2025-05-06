# Thai Sentiment Classification with Explainable AI

This project is a practical implementation of a sentiment classification system for Thai-language social media messages using the **Wisesight Sentiment Corpus**. It applies **transformer-based models** (e.g., WangchanBERTa) and incorporates **LIME** for explainable AI to interpret and visualize predictions.

---

## 1. Project Objective
- Classify Thai text into four sentiment categories: `positive`, `neutral`, `negative`, `question`
- Provide interpretable output that highlights which tokens influence model predictions
- Enable experimentation with real-world conversational data in Thai

---

## 2. Dataset Overview
- **Source:** Wisesight Sentiment Corpus (26,737 entries)
- **Fields:**
  - `texts`: Thai language text
  - `category`: One of `pos`, `neu`, `neg`, or `q`
- **Label Mapping:**
  - `pos` → 0 (Positive)
  - `neu` → 1 (Neutral)
  - `neg` → 2 (Negative)
  - `q`   → 3 (Question)
- **Split:**
  - Train: 21,628 samples
  - Validation: 2,404 samples
  - Test: 2,671 samples

---

## 3. Preprocessing Pipeline
- Clean and normalize Thai text
- Tokenize using **PyThaiNLP** (`newmm` engine)
- Encode target labels using `LabelEncoder`
- Prepare input sequences with proper truncation and padding for transformer models

---

## 4. Model Training
- **Base Model:** `airesearch/wangchanberta-base-att-spm-uncased`
- **Framework:** Hugging Face Transformers
- **Trainer API** used for model fine-tuning
- Evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score (macro and per-class)
  - Confusion matrix visualization

---

## 5. Explainability with LIME
- **LIME Integration:**
  - Wrap Hugging Face classifier for compatibility with `LimeTextExplainer`
  - Use LIME to interpret predictions by highlighting token contributions
- **Output Examples:**
  - Bar charts showing most influential tokens per class
  - Analysis of misclassified cases vs correct ones

---

## 6. Optional Deployment
- Build a **Streamlit** interface:
  - Input Thai message → Predict sentiment → Show LIME explanation
  - CSV batch mode for sentiment review

---

## 7. Environment Setup
```bash
pip install transformers datasets scikit-learn pythainlp lime pandas matplotlib seaborn
```

---

## 8. Author
**Worawat Kaewsanmuang**  
[GitHub](https://github.com/lottettt) • [LinkedIn](https://linkedin.com/in/worawat)

---

## 9. License
Dataset released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
