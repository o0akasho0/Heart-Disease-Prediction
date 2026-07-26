# ❤️ Heart Disease Prediction using Machine Learning

A Streamlit web app that explores a heart disease dataset, trains three
classification models (Logistic Regression, Decision Tree, Random Forest),
and lets a user enter patient details to get a live prediction.

---

## 🚀 Live Demo
Deployed on Streamlit Community Cloud: **https://heart-disease-prediction-akash0.streamlit.app/***

---

## 📂 Project Structure
```
├── heart_disease_fixed.py   # Main Streamlit app
├── heart.csv                # Dataset used for training
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🧠 What the App Does

1. **Data Overview** – shows the raw dataset, shape, missing values, and duplicates.
2. **Exploratory Data Analysis (EDA)** – visualizes target distribution, gender vs
   disease, correlation heatmap, chest pain type vs disease, age vs max heart
   rate, and cholesterol distribution.
3. **Preprocessing** – one-hot encodes categorical columns (`cp`, `restecg`,
   `slope`, `thal`) and scales numeric features with `StandardScaler`.
4. **Model Training & Evaluation** – trains Logistic Regression, Decision
   Tree, and Random Forest; reports Accuracy, Precision, Recall, F1,
   Confusion Matrix, and ROC-AUC.
5. **Model Comparison** – bar chart comparing accuracy across the 3 models.
6. **Feature Importance** – top 10 most important features from the Random
   Forest model.
7. **Live Prediction** – enter patient details in the form and get an instant
   prediction (No Disease / Disease Detected) with class probabilities.

---

## 📊 Dataset

The dataset (`heart.csv`) contains the following columns:

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | 0 = Female, 1 = Male |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar > 120 mg/dl (0/1) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (0/1) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia (0–3) |
| `target` | 0 = No Disease, 1 = Disease |

> **Note:** The dataset used here was curated/generated so that risk factors
> behave in a medically intuitive direction (e.g. higher cholesterol, blood
> pressure, and blocked vessels increase predicted risk). If you swap in a
> different `heart.csv` with the same column names, the app will retrain
> automatically — no code changes needed.

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** – web app framework
- **pandas / numpy** – data handling
- **matplotlib / seaborn** – visualizations
- **scikit-learn** – ML models (Logistic Regression, Decision Tree, Random Forest)

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run heart_disease_fixed.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this project (including `heart.csv` and `requirements.txt`) to a
   GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select your repo, branch, and set the main file
   path to `heart_disease_fixed.py`.
4. Click **Deploy** — the app will be live in a couple of minutes.

---

## 🔮 Example Predictions

**Likely "No Heart Disease":**
Age 25, Female, CP 0, BP 110, Chol 150, FBS 0, RestECG 0, Max HR 190,
Exang 0, Oldpeak 0.0, Slope 0, Vessels 0, Thal 1

**Likely "Heart Disease Detected":**
Age 65, Male, CP 3, BP 180, Chol 350, FBS 1, RestECG 2, Max HR 95,
Exang 1, Oldpeak 4.5, Slope 2, Vessels 4, Thal 3

---

## ⚠️ Disclaimer
This project is for **educational purposes only** and is **not** a
substitute for professional medical diagnosis.

---

## 👤 Author
Akash kumar Gupta

