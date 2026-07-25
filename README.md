# ❤️ Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease using patient health data. The project covers full EDA, data preprocessing, model training, evaluation, and an interactive prediction feature for new patient input.

## 📊 Dataset

The dataset used is the **Heart Disease UCI dataset** (`heart.csv`), containing patient clinical features such as age, sex, chest pain type, cholesterol, resting blood pressure, and more.

> Note: If the dataset file is large or you'd rather not upload raw data, you can link the source here instead, e.g. [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets).

## 🔍 Project Workflow

1. **Data Loading & Cleaning** — loading `heart.csv`, checking nulls, removing duplicates
2. **Exploratory Data Analysis (EDA)**
   - Target distribution
   - Heart disease by gender
   - Correlation heatmap
   - Chest pain type vs heart disease
   - Age vs maximum heart rate
   - Cholesterol distribution
3. **Feature Engineering** — One-hot encoding of categorical features (`cp`, `restecg`, `slope`, `thal`)
4. **Model Training**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix
   - ROC-AUC Score & ROC Curve
6. **Model Comparison** — Accuracy comparison across all three models
7. **Feature Importance** — Identifying top health indicators (Random Forest + Logistic Regression coefficients)
8. **Live Prediction** — Takes patient details as input and predicts heart disease risk

## 🏆 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | *fill in your value* |
| Decision Tree | *fill in your value* |
| Random Forest | *fill in your value* |

## 🩺 Top Health Indicators

1. Chest Pain Type (`cp`)
2. Maximum Heart Rate (`thalach`)
3. ST Depression (`oldpeak`)
4. Number of Major Vessels (`ca`)
5. Thalassemia (`thal`)

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/o0akasho0/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the script
python heart_disease_prediction.py
```

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart.csv                     # Dataset
├── heart_disease_prediction.py   # Main script
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## 📌 Future Improvements

- Deploy as a web app using Streamlit or Flask
- Hyperparameter tuning for better accuracy
- Add cross-validation
- Try additional models (XGBoost, SVM)

## 🤝 Contributing

Feel free to fork this repo and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.
