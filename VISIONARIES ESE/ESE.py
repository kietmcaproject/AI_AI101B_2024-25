import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)
from sklearn.utils import resample

# Load Dataset
try:
    df = pd.read_csv('C:\\Users\\devansh\\OneDrive\\Documents\\Visual code\\Python\\AI\\creditcard.csv')
except FileNotFoundError:
    print("Make sure 'creditcard.csv' is uploaded.")

# Data Overview
print("Dataset Info:")
df.info()

print("\nDataset Head:")
print(df.head())

print("\nClass Distribution:")
print(df['Class'].value_counts())

# Plot Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Non-Fraudulent, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Preprocessing
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# Upsampling (with 30% of majority class to reduce time)
X_train_majority = X_train[y_train == 0]
y_train_majority = y_train[y_train == 0]
X_train_minority = X_train[y_train == 1]
y_train_minority = y_train[y_train == 1]

n_samples = int(len(X_train_majority) * 0.3)
X_train_minority_upsampled, y_train_minority_upsampled = resample(
    X_train_minority, y_train_minority,
    replace=True,
    n_samples=n_samples,
    random_state=42
)

X_train_majority_sampled = X_train_majority.sample(n=n_samples, random_state=42)
y_train_majority_sampled = y_train_majority.loc[X_train_majority_sampled.index]

X_train_upsampled = pd.concat([X_train_majority_sampled, X_train_minority_upsampled])
y_train_upsampled = pd.concat([y_train_majority_sampled, y_train_minority_upsampled])

print("\nUpsampled training class distribution:")
print(y_train_upsampled.value_counts())

# Logistic Regression
start = time.time()
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_upsampled, y_train_upsampled)
print(f"Logistic Regression training time: {time.time() - start:.2f}s")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Evaluation ---")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC (Logistic Regression): {roc_auc:.4f}")

plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Logistic Regression)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Random Forest (optimized)
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=50,      # Reduced trees
    max_depth=12,         # Limit tree depth
    max_features='sqrt',  # Faster splits
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_upsampled, y_train_upsampled)
print(f"Random Forest training time: {time.time() - start:.2f}s")

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Evaluation ---")
print(classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(f"AUC (Random Forest): {roc_auc_rf:.4f}")

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Random Forest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Final Comparison Table
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n--- Final Comparison ---")
print(f"{'Metric':<18} | {'Logistic Regression':<20} | {'Random Forest':<15}")
print(f"{'-'*18}-+-{'-'*20}-+-{'-'*15}")
print(f"{'Precision (Fraud)':<18} | {precision_lr:<20.4f} | {precision_rf:<15.4f}")
print(f"{'Recall (Fraud)':<18} | {recall_lr:<20.4f} | {recall_rf:<15.4f}")
print(f"{'F1-Score (Fraud)':<18} | {f1_lr:<20.4f} | {f1_rf:<15.4f}")
print(f"{'AUC':<18} | {roc_auc:<20.4f} | {roc_auc_rf:<15.4f}")
