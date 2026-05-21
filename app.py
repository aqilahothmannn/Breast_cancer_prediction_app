import numpy as np
import pandas as pd
import gradio as gr
import pickle

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=250
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
}

# Train and evaluate all models
results = []
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy*100:.2f}%")
    results.append({"Model": name, "Accuracy": accuracy})

# Pick the best model
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

best_model_name = df_results.loc[0, "Model"]
best_model = models[best_model_name]

# ✅ SAVE the best model as pkl
with open("DT_cancer.pkl", "wb") as file:
    pickle.dump(best_model, file)

print(f"Best model: {best_model_name} saved as DT_cancer.pkl")

feature_names = data.feature_names

def predict(*inputs):
    input_array = np.array(list(inputs)).reshape(1, -1)
    prediction = best_model.predict(input_array)[0]
    probability = best_model.predict_proba(input_array)[0]
    label = data.target_names[prediction]
    return f"{label} (Malignant: {probability[0]:.2%}, Benign: {probability[1]:.2%})"

inputs = [
    gr.Number(label=name, value=float(X_test[0][i]))
    for i, name in enumerate(feature_names)
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Text(label="Prediction"),
    title="🎗️ Breast Cancer Classifier",
    description=f"Best Model: {best_model_name}, Accuracy: {df_results.loc[0, 'Accuracy']*100:.2f}%"
)

demo.launch(share=True)