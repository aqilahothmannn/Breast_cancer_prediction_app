# 🎗️ Breast Cancer Predictor

An interactive machine learning web app that predicts whether a breast tumor is **Malignant** or **Benign** based on medical measurements — built with Gradio and Scikit-learn.

🔗 **Live Demo:** [Try it on Hugging Face](https://huggingface.co/spaces/aqilahothman/breastcancer)

---

## 📌 About The Project

This app uses the **Wisconsin Breast Cancer Dataset** to train and compare multiple machine learning models. The best performing model is automatically selected and used for predictions.

The app takes 30 medical features as input and returns:
- The predicted class (**Malignant** or **Benign**)
- The probability score for each class

---

## 🤖 Models Compared

| Model | Accuracy |
|---|---|
| Logistic Regression | 93.86% |
| Decision Tree | 88.60% |
| **Random Forest** ✅ | **94.74%** |

> Random Forest was selected as the best model automatically.

---

## 🚀 Features

- Compares 3 ML models and picks the best one automatically
- Interactive UI built with Gradio
- Shows prediction with probability scores
- Uses the real Wisconsin Breast Cancer Dataset from Scikit-learn

---

## 🛠️ Tech Stack

- **Python**
- **Gradio** — for the web interface
- **Scikit-learn** — for ML models and dataset
- **NumPy & Pandas** — for data processing

---

## ⚙️ How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/aqilahothmannn/breast-cancer-predictor.git
cd breast-cancer-predictor
```

2. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
python app.py
```

5. Open your browser and go to `http://localhost:7860`

---

## 📦 Requirements

```
numpy
pandas
gradio
scikit-learn
```

---

## 📊 Dataset

- **Name:** Wisconsin Breast Cancer Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:** 30 numeric medical measurements
- **Target:** Malignant (0) or Benign (1)

---

## 👩‍💻 Author

**Aqilah Othman**
- GitHub: [@aqilahothmannn](https://github.com/aqilahothmannn)
- Hugging Face: [@aqilahothman](https://huggingface.co/aqilahothman)