# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load the Data ---
df = pd.read_excel('/Users/sujiththota/Downloads/Python/Research/ML_DATA/cleaned_merged_output.xlsx')  # Path to cleaned file

# --- Inspect the Data ---
print(df.columns)  # Check all columns

# --- Feature Selection ---
# Drop non-numeric or irrelevant columns
# (Assuming 'Depressed post' is the label, drop any IDs, text, image/video names)
drop_columns = ['Media Name', 'Profile Name', 'Simple Description', 'Embedded Text', 'Caption', 'Important Note', 'Diagnosed Date', 'Media Type']

# Drop if these exist in your columns
drop_columns = [col for col in drop_columns if col in df.columns]

# Define features
X = df.drop(columns=drop_columns + ['Depressed post'], errors='ignore')

# Only keep numeric columns
X = X.select_dtypes(include=['number'])

# Define label
y = df['Depressed post'].map({'Yes': 1, 'No': 0})

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# --- Predictions ---
y_pred_rf = rf_model.predict(X_test)

# --- Evaluation ---

# Accuracy
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest (All Features)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Feature Importance ---
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='mako')
plt.title('Feature Importance - Random Forest (All Features)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()
