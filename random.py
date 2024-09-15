import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load the data
train_data = pd.read_csv('C:/Users/Sidharth/Desktop/AI&ML/carvan_train (1).csv')
test_data = pd.read_csv('C:/Users/Sidharth/Desktop/AI&ML/carvan_test (2).csv')

# Extract the target variable
X = train_data.drop(columns=['V86'])
y = train_data['V86']

# Data Preprocessing
# 1. Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
test_data = imputer.transform(test_data)

# 2. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data = scaler.transform(test_data)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the resampled data for validation
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model Training with RandomForest
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_

# Validate the model
y_val_pred_proba = best_rf_model.predict_proba(X_val)[:, 1]
threshold = 0.3  # Adjusting the threshold
y_val_pred = (y_val_pred_proba >= threshold).astype(int)

# Print accuracy and classification report
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
print(classification_report(y_val, y_val_pred, zero_division=0))

# Calculate and print ROC AUC score
roc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation ROC AUC Score: {roc_score}")

# Calculate and print F-beta score (you can adjust beta as needed)
beta = 0.5  # Example: more weight on precision
fbeta = fbeta_score(y_val, y_val_pred, beta=beta, zero_division=0)
print(f"Validation F-beta Score (beta={beta}): {fbeta}")

# Predict on the test set using the adjusted threshold
y_test_pred_proba = best_rf_model.predict_proba(test_data)[:, 1]
y_test_pred = (y_test_pred_proba >= threshold).astype(int)

# Save the predictions to a CSV file
submission = pd.DataFrame({'Customer_ID': range(1, len(y_test_pred) + 1), 'V86': y_test_pred})
submission.to_csv('submission.csv', index=False)

