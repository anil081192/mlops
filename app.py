import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import dagshub

# Load the data
data = pd.read_csv(r'C:\Users\gadea\Desktop\practice\mlops\data\WineQT.csv')

# Assume 'quality' is the target column
# For classification, let's say quality >= 7 is 'good' (1), else 'bad' (0)
data['quality_label'] = (data['quality'] >= 7).astype(int)

# Features and target
feature_columns = [col for col in data.columns if col not in ['quality', 'quality_label']]
X = data[feature_columns]
y = data['quality_label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['bad', 'good']))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

import mlflow.sklearn
import dagshub
import os
dagshub.init(repo_owner="anil081192", repo_name="my-first-repo", mlflow=True)
with mlflow.start_run():
    mlflow.sklearn.log_model(clf, "random_forest_model")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.set_tag("author", "anil")
    # Log artifacts (e.g., classification report and feature importance)
    # Save classification report
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=['bad', 'good']))
    mlflow.log_artifact(report_path)

    # Save feature importances
    importances_path = "feature_importances.csv"
    feature_importances = pd.DataFrame({
        "feature": X.columns,
        "importance": clf.feature_importances_
    }).sort_values(by="importance", ascending=False)
    feature_importances.to_csv(importances_path, index=False)
    mlflow.log_artifact(importances_path)

