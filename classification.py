import pandas as pd
import numpy as np

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# K-Nearest Neighbors classifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

#============================
# Random Forest Classification
#===========================

# 1. Load dataset
df = pd.read_csv("dataset_features_ml.csv", sep=";")

X = df[["ambient_temp_C", "ambient_slope", "ambient_rolling_mean"]]
y = df["class"]   # 0 / 1 / 2

# 2. Train / test split (70 / 30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y   # IMPORTANT : équivalent propre
)

# 3. Random Forest (avec OOB)
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True
)

rf.fit(X_train, y_train)

# 4. Test set evaluation
y_pred = rf.predict(X_test)

conf_test = confusion_matrix(y_test, y_pred)
acc_test = accuracy_score(y_test, y_pred)
err_test = 1 - acc_test

print("Confusion matrix (test set):")
print(conf_test)

print(f"\nTest accuracy: {acc_test:.3f}")
print(f"Test error: {err_test:.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 5. OOB error (équivalent R)
acc_oob = rf.oob_score_
err_oob = 1 - acc_oob

print(f"OOB accuracy: {acc_oob:.3f}")
print(f"OOB error: {err_oob:.3f}")

# 6. Variable importance
importances = rf.feature_importances_
feature_names = X.columns

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nVariable importance:")
print(imp_df)


#============================
# K-Nearest Neighbors Classification   
#============================

# 7. Feature scaling (required for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. KNN model definition
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)

# 9. Model training
knn.fit(X_train_scaled, y_train)

# 10. Test set evaluation
y_pred_knn = knn.predict(X_test_scaled)

conf_test_knn = confusion_matrix(y_test, y_pred_knn)
acc_test_knn = accuracy_score(y_test, y_pred_knn)
err_test_knn = 1 - acc_test_knn

print("\nKNN - Confusion matrix (test set):")
print(conf_test_knn)

print(f"\nKNN - Test accuracy: {acc_test_knn:.3f}")
print(f"KNN - Test error: {err_test_knn:.3f}")

print("\nKNN - Classification report:")
print(classification_report(y_test, y_pred_knn))