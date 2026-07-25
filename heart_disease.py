import pandas as pd
import matplotlib.pyplot as plt #type:ignore
import seaborn as sns   #type:ignore

from sklearn.model_selection import train_test_split  #type:ignore
from sklearn.preprocessing import StandardScaler  #type:ignore
from sklearn.linear_model import LogisticRegression  #type:ignore
from sklearn.tree import DecisionTreeClassifier  #type:ignore
from sklearn.ensemble import RandomForestClassifier  #type:ignore


from sklearn.metrics import (  #type:ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Load Dataset

df = pd.read_csv("heart.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nStatistical Summary")
print(df.describe())
print("\nMissing Values")
print(df.isnull().sum())
print("\nDuplicate Values")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")
print(df.shape)
#EDA
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)

plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="sex", hue="target", data=df)
plt.title("Heart Disease by Gender")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="cp", hue="target", data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="age", y="thalach", hue="target", data=df)
plt.title("Age vs Maximum Heart Rate")
plt.show()

plt.figure(figsize=(6,4))

sns.histplot(df["chol"], bins=20, kde=True)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.show()


df = pd.get_dummies(   #ONEHOTENCODING
    df,
    columns=["cp", "restecg", "slope", "thal"],
    drop_first=True
)

print("\nDataset After One-Hot Encoding")
print(df.head())

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]
print("\nFeature Names")
print(X.columns)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy :", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision :", precision)

recall = recall_score(y_test, y_pred)
print("Recall :", recall)

f1 = f1_score(y_test, y_pred)
print("F1 Score :", f1)

#cnfuson matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score :", roc_auc)

fpr, tpr, threshold = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0,1],[0,1],"r--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy :", dt_accuracy)

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy :", rf_accuracy)

# Model Comparison

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

accuracies = [
    accuracy,
    dt_accuracy,
    rf_accuracy
]

comparison = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies
})

print("\nModel Comparison")
print(comparison)

plt.figure(figsize=(8,5))  #model comparison graph
sns.barplot(
    x="Model",
    y="Accuracy",
    data=comparison
)

plt.title("Model Comparison")
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")  #feature importance
print(feature_importance)

plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)
plt.title("Top Health Indicators")
plt.show()


# Logistic Regression Feature Importance
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute Value"] = coefficients["Coefficient"].abs()

coefficients = coefficients.sort_values(
    by="Absolute Value",
    ascending=False
)

print("\nLogistic Regression Feature Importance")
print(coefficients)
print("\nTop Health Indicators (Plain Language):")
print("1. Chest Pain Type (cp) is one of the strongest indicators of heart disease.")
print("2. Maximum Heart Rate (thalach) has a strong relationship with heart disease.")
print("3. Oldpeak (ST depression) helps identify patients at higher risk.")
print("4. Number of Major Vessels (ca) significantly affects prediction.")
print("5. Thal is an important clinical feature for heart disease prediction.")

# User Prediction
print("\nEnter Patient Details")
age = int(input("Age : "))
sex = int(input("Sex (0=Female, 1=Male) : "))
cp = int(input("Chest Pain Type (0-3) : "))
trestbps = int(input("Resting Blood Pressure : "))
chol = int(input("Cholesterol : "))
fbs = int(input("Fasting Blood Sugar (0/1) : "))
restecg = int(input("Rest ECG (0-2) : "))
thalach = int(input("Maximum Heart Rate : "))
exang = int(input("Exercise Induced Angina (0/1) : "))
oldpeak = float(input("Oldpeak : "))
slope = int(input("Slope (0-2) : "))
ca = int(input("Number of Major Vessels (0-4) : "))
thal = int(input("Thal (1-3) : "))

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

user_df = pd.DataFrame([user])
user_df = user_df.reindex(columns=X.columns, fill_value=0)
user_scaled = scaler.transform(user_df)
prediction = model.predict(user_scaled)
if prediction[0] == 1:
    print("\nPrediction : Heart Disease Detected")
else:
    print("\nPrediction : No Heart Disease")