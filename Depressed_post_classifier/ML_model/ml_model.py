# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Load the Data ---
df = pd.read_excel('/Users/sujiththota/Downloads/Python/Research/ML_DATA/cleaned_merged_output.xlsx')

# --- Feature Selection ---
drop_columns = ['Media Name', 'Profile Name', 'Simple Description', 'Embedded Text', 'Caption', 'Important Note', 'Diagnosed Date', 'Media Type']
drop_columns = [col for col in drop_columns if col in df.columns]

X = df.drop(columns=drop_columns + ['Depressed post'], errors='ignore')
X = X.select_dtypes(include=['number'])

# --- Fill Missing Values ---
X = X.fillna(X.mean())  # Fill NaNs with column mean

# Define label
y = df['Depressed post'].map({'Yes': 1, 'No': 0})

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling (important for Logistic Regression) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression Model ---
logreg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# --- Predictions ---
y_pred_logreg = logreg_model.predict(X_test_scaled)

# --- Evaluation ---

# Accuracy
accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logreg))

# Confusion Matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix - Logistic Regression (All Features)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Feature Importance (Coefficients) ---
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(logreg_model.coef_[0])
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='rocket')
plt.title('Feature Importance - Logistic Regression (All Features)')
plt.xlabel('Importance (Absolute Coefficient)')
plt.ylabel('Feature')
plt.show()
