import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction using Machine Learning")
st.write("Logistic Regression | Decision Tree | Random Forest")

CAT_COLS = ["cp", "restecg", "slope", "thal"]


# ---------------------------------------------------------------
# Cache data + model training so it does NOT retrain on every
# widget interaction / button click (this was silently happening
# in the original script since the whole file re-ran each time).
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    raw_df = pd.read_csv("heart.csv")
    return raw_df


@st.cache_resource
def train_models(raw_df: pd.DataFrame):
    df = raw_df.drop_duplicates().copy()

    # Make sure categorical columns are plain integers BEFORE one-hot
    # encoding. If any of these ever contain NaN, pandas silently
    # upcasts the column to float, and get_dummies would then create
    # columns like "cp_1.0" instead of "cp_1" -- this was the actual
    # bug causing every prediction to come out the same, because the
    # manually-built one-hot columns in the old code never matched.
    for c in CAT_COLS:
        df[c] = df[c].astype(int)

    encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    X = encoded.drop("target", axis=1)
    y = encoded["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    return {
        "df": df,
        "encoded": encoded,
        "X": X,
        "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "log_model": log_model,
        "dt_model": dt_model,
        "rf_model": rf_model,
    }


raw_df = load_data()
state = train_models(raw_df)

df = state["df"]
X = state["X"]
y = state["y"]
X_test_scaled = state["X_test_scaled"]
y_test = state["y_test"]
scaler = state["scaler"]
model = state["log_model"]           # Logistic Regression = main model used for prediction
dt_model = state["dt_model"]
rf_model = state["rf_model"]

st.subheader("Dataset")
st.dataframe(df.head())
st.write("Shape :", df.shape)
st.write("Missing Values")
st.write(df.isnull().sum())
st.write("Duplicate Values :", raw_df.duplicated().sum())


# Exploratory Data Analysis
st.header("Exploratory Data Analysis")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="target", data=df, ax=ax)
ax.set_title("Target Distribution")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="sex", hue="target", data=df, ax=ax)
ax.set_title("Heart Disease by Gender")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="cp", hue="target", data=df, ax=ax)
ax.set_title("Chest Pain Type vs Heart Disease")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x="age", y="thalach", hue="target", data=df, ax=ax)
ax.set_title("Age vs Maximum Heart Rate")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["chol"], bins=20, kde=True, ax=ax)
ax.set_title("Cholesterol Distribution")
st.pyplot(fig)

st.subheader("Dataset After One Hot Encoding")
st.dataframe(state["encoded"].head())
st.write("Feature columns used by the model:", X.columns.tolist())


# Model Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.header("Model Evaluation")
st.write("### Logistic Regression")
st.write("Accuracy :", accuracy)
st.write("Precision :", precision)
st.write("Recall :", recall)
st.write("F1 Score :", f1)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

roc_auc = roc_auc_score(y_test, y_prob)
st.write("ROC-AUC Score :", roc_auc)

fpr, tpr, threshold = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fpr, tpr, label="Logistic Regression")
ax.plot([0, 1], [0, 1], "r--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

dt_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)
st.write("Decision Tree Accuracy :", dt_accuracy)

rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
st.write("Random Forest Accuracy :", rf_accuracy)

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy, dt_accuracy, rf_accuracy]
})
st.subheader("Model Comparison")
st.dataframe(comparison)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=comparison, ax=ax)
ax.set_title("Model Comparison")
st.pyplot(fig)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Top 10 Important Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10), ax=ax)
ax.set_title("Top Health Indicators")
st.pyplot(fig)


# ---------------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------------
st.header("Heart Disease Prediction")

age = st.number_input("Age", 1, 100, 30)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    # Manual one-hot encoding (matches how training data was one-hot
    # encoded with pd.get_dummies(..., drop_first=True) on cp/restecg/
    # slope/thal). Since CAT_COLS were forced to int during training
    # (see train_models), the dummy column names are guaranteed to be
    # "cp_1", "cp_2", ... and NOT "cp_1.0" -- so these manual 0/1
    # flags will always match the columns the model was trained on.
    user = {
        "age": age,
        "sex": sex,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "ca": ca,

        "cp_1": 0,
        "cp_2": 0,
        "cp_3": 0,

        "restecg_1": 0,
        "restecg_2": 0,

        "slope_1": 0,
        "slope_2": 0,

        "thal_1": 0,
        "thal_2": 0,
        "thal_3": 0
    }

    if cp == 1:
        user["cp_1"] = 1
    elif cp == 2:
        user["cp_2"] = 1
    elif cp == 3:
        user["cp_3"] = 1

    if restecg == 1:
        user["restecg_1"] = 1
    elif restecg == 2:
        user["restecg_2"] = 1

    if slope == 1:
        user["slope_1"] = 1
    elif slope == 2:
        user["slope_2"] = 1

    if thal == 1:
        user["thal_1"] = 1
    elif thal == 2:
        user["thal_2"] = 1
    elif thal == 3:
        user["thal_3"] = 1

    user_encoded = pd.DataFrame([user]).reindex(columns=X.columns, fill_value=0)

    with st.expander("Debug: encoded input sent to the model"):
        st.dataframe(user_encoded)

    user_scaled = scaler.transform(user_encoded)
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)[0]

    st.write("Prediction Value:", prediction[0])
    st.write(f"Probability -> No Disease: {probability[0]:.3f} | Disease: {probability[1]:.3f}")

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")

