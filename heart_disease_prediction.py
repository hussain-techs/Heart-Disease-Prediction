# ============================================
# Task 3: Heart Disease Prediction
# Complete Program
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ============================================
# Upload Dataset (For Google Colab)
# ============================================

from google.colab import files

uploaded = files.upload()

# ============================================
# Load Dataset
# ============================================

# Make sure your file name is heart.csv
df = pd.read_csv("heart.csv")

# ============================================
# Display Dataset Information
# ============================================

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# ============================================
# Check Missing Values
# ============================================

print("\nMissing Values:\n")
print(df.isnull().sum())

# Fill Missing Values
df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)

# ============================================
# Exploratory Data Analysis (EDA)
# ============================================

# Heart Disease Distribution
plt.figure(figsize=(6,4))

df['target'].value_counts().plot(kind='bar')

plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Count")

plt.show()

# ============================================
# Correlation Heatmap
# ============================================

plt.figure(figsize=(12,8))

correlation = df.corr()

plt.imshow(correlation, cmap='coolwarm', aspect='auto')

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================
# Age vs Heart Disease
# ============================================

plt.figure(figsize=(6,4))

plt.scatter(df['age'], df['target'])

plt.xlabel("Age")
plt.ylabel("Heart Disease")
plt.title("Age vs Heart Disease")

plt.show()

# ============================================
# Feature Selection
# ============================================

X = df.drop('target', axis=1)

y = df['target']

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Feature Scaling
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================
# Train Logistic Regression Model
# ============================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ============================================
# Predictions
# ============================================

y_pred = model.predict(X_test)

# Probability Scores
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================
# Model Evaluation
# ============================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:")
print(roc_score)

# ============================================
# ROC Curve
# ============================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, label="ROC Curve")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.show()

# ============================================
# Feature Importance
# ============================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nImportant Features:\n")
print(importance)

# ============================================
# Feature Importance Graph
# ============================================

plt.figure(figsize=(10,6))

plt.barh(
    importance['Feature'],
    importance['Importance']
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Heart Disease Prediction")

plt.show()

# ============================================
# End of Program
# ============================================
