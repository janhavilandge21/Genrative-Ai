
# 🏭 AI-Powered Pharma Manufacturing Intelligence System

An end-to-end AI system designed to improve pharmaceutical manufacturing by predicting batch failures, detecting anomalies, and generating intelligent recommendations using Machine Learning and Generative AI.

---

## 🚀 Project Overview

In pharmaceutical manufacturing, batch failures can lead to significant financial losses and delays. This project leverages Machine Learning and Generative AI to:

* Predict batch failure probability
* Detect abnormal production patterns
* Provide AI-driven recommendations for process optimization

---

## 🧠 Key Features

✅ Batch Failure Prediction using XGBoost
✅ Anomaly Detection using Isolation Forest
✅ Explainability using Feature Importance & SHAP
✅ AI-generated Recommendations using Groq LLM (LLaMA 3.1)
✅ Interactive UI using Streamlit

---

## 🏗️ Architecture

* **Data Layer:** Synthetic pharma manufacturing dataset
* **ML Layer:** XGBoost classifier + Isolation Forest
* **Explainability:** SHAP values
* **Gen AI Layer:** Groq API (LLaMA 3.1)
* **Frontend:** Streamlit dashboard

---

## 📊 Input Parameters

* Temperature
* Pressure
* Mixing Time
* Material Quality Score

---

## 📈 Output

* Failure Probability (%)
* Anomaly Detection Result
* AI-generated Explanation & Recommendations

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Groq API (LLaMA 3.1)
* Streamlit

---

## ⚡ Installation

```bash
git clone https://github.com/your-username/pharma-ai-system.git
cd pharma-ai-system
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

---

## ▶️ Run Training Pipeline

```bash
python train_pipeline.py
```

---

## 🎯 Run Streamlit App

```bash
cd frontend
streamlit run app.py
```

---

## 📸 Demo

* Input batch parameters
* Get failure prediction
* View anomaly status
* Receive AI-based recommendations

---

## 💡 Use Cases

* Pharma manufacturing quality control
* Process optimization
* Predictive maintenance
* Production monitoring

---

## 🧠 AI Explanation (Example)

> “The batch has a high probability of failure due to low material quality and elevated temperature. It is recommended to improve raw material consistency and reduce temperature to ensure stability.”

---

## 📊 Model Performance

* Accuracy: XX%
* ROC-AUC: XX
* Precision: XX
* Recall: XX

---


## 🧑‍💻 Author

**Janhavi Landge**

* LinkedIn: https://www.linkedin.com/in/janhavilandge-datascience/
* GitHub: https://github.com/janhavilandge21

---

## ⭐ Future Improvements

* Real-time data streaming integration
* Advanced deep learning models
* Deployment using Docker & AWS
* CI/CD pipeline integration

---

## 📌 Conclusion

This project demonstrates how AI can transform pharmaceutical manufacturing by improving efficiency, reducing failures, and enabling data-driven decision-making.

---
