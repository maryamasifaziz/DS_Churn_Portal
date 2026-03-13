# 🏦 Customer Churn Prediction

> **A deep learning-powered web app that predicts whether a bank customer is likely to churn based on their profile — built with TensorFlow, Scikit-learn, and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-green)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![Dataset](https://img.shields.io/badge/Dataset-10K%20Customers-purple)

---

## 🚀 Live Demo
🔗 [View Live App](https://ds-customer-churn-prediction.streamlit.app)

---

## 📸 Screenshots
![Dashboard Screenshot 1](images/dashboard.png)

---

## 🧠 Overview
This app predicts **customer churn** for a bank using a trained **Neural Network (ANN)** model. Enter a customer's details and instantly get the churn probability — helping banks proactively retain at-risk customers before they leave.

---

## ✨ Features
- Real-time churn probability prediction
- Interactive input sliders and dropdowns
- Handles geography (France, Spain, Germany) via One-Hot Encoding
- Gender encoding via Label Encoder
- Feature scaling with StandardScaler
- Clear prediction output with probability score

---

## 🏗️ Model Pipeline

```
Raw Customer Data (10K records)
        ↓
Label Encoding (Gender)
        ↓
One-Hot Encoding (Geography)
        ↓
StandardScaler (Feature Scaling)
        ↓
ANN Model (TensorFlow/Keras)
        ↓
Churn Probability (0–1)
```

---

## 📊 Input Features

| Feature | Description |
|---|---|
| Credit Score | Customer's credit score (300–1000) |
| Geography | Country — France, Spain, Germany |
| Gender | Male / Female |
| Age | Customer age (16–69) |
| Tenure | Years with the bank (0–10) |
| Balance | Account balance |
| Number of Products | Products used (1–4) |
| Has Credit Card | Yes / No |
| Is Active Member | Yes / No |
| Estimated Salary | Annual salary estimate |

---

## 🧰 Tech Stack
| Tool | Purpose |
|---|---|
| TensorFlow / Keras | ANN model training + inference |
| Scikit-learn | Label encoding, one-hot encoding, scaling |
| Pandas + NumPy | Data preprocessing |
| Streamlit | Interactive web interface |

---

## 📁 Project Structure
```
customer-churn-prediction/
├── app.py                      # Streamlit app
├── model.h5                    # Trained ANN model
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Fitted LabelEncoder
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder
├── Churn_Modelling.csv         # Dataset (10K customers)
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally
```bash
git clone https://github.com/maryamasifaziz/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Dataset
- **Source:** Churn Modelling Dataset
- **Size:** 10,000 customer records
- **Target:** `Exited` — 1 (churned) or 0 (retained)
- **Features:** Credit score, geography, gender, age, tenure, balance, products, salary

---

## 👤 Author
**Maryam Asif**  
🎓 FAST NUCES    
🔗 [LinkedIn](https://linkedin.com/maryamasifaziz) | [GitHub](https://github.com/maryamasifaziz)
