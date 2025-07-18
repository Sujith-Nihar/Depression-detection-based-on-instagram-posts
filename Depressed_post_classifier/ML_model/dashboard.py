# Save this as app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- Title and Summary ---
st.title("Depression Detection Dashboard")

st.subheader("📚 Project Summary")
st.markdown("""
This project aims to classify social media posts as **Depressed** or **Not Depressed** based on visual features 
like **Brightness**, **Saturation**, **Hue** along with **Emotion Scores** and **PHQ-9 Features** extracted from captions.
We trained models like **Logistic Regression** and **Random Forest** and evaluated their performance.

The goal was to see if image properties and textual emotional scores correlate with signs of depression.
""")

# --- Load Data ---
st.subheader("🗂️ Data Overview")
df = pd.read_excel('/Users/sujiththota/Downloads/Python/Research/ML_DATA/cleaned_merged_output.xlsx')
st.dataframe(df.head())

# --- Feature Selection ---
drop_columns = ['Media Name', 'Profile Name', 'Simple Description', 'Embedded Text', 'Caption', 'Important Note', 'Diagnosed Date', 'Media Type']
drop_columns = [col for col in drop_columns if col in df.columns]

X = df.drop(columns=drop_columns + ['Depressed post'], errors='ignore')
X = X.select_dtypes(include=['number']).fillna(X.mean())

y = df['Depressed post'].map({'Yes': 1, 'No': 0})

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training (Random Forest) ---
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# --- Model Training (Logistic Regression) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)
y_pred_logreg = logreg_model.predict(X_test_scaled)

# --- Visualization Section ---
st.subheader("📈 Visualizations")

# Confusion Matrix Random Forest
st.markdown("**Random Forest - Confusion Matrix**")
cm_rf = confusion_matrix(y_test, y_pred_rf)
fig_rf, ax_rf = plt.subplots()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax_rf)
st.pyplot(fig_rf)

# Confusion Matrix Logistic Regression
st.markdown("**Logistic Regression - Confusion Matrix**")
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
fig_logreg, ax_logreg = plt.subplots()
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Purples', ax=ax_logreg)
st.pyplot(fig_logreg)

# Feature Importances Random Forest
st.markdown("**Random Forest - Feature Importance**")
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig_feat, ax_feat = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='mako', ax=ax_feat)
st.pyplot(fig_feat)

# --- Model Evaluation ---
st.subheader("📝 Model Performance")

# Accuracy
st.write(f"**Random Forest Accuracy:** {round(rf_model.score(X_test, y_test), 4)}")
st.write(f"**Logistic Regression Accuracy:** {round(logreg_model.score(X_test_scaled, y_test), 4)}")

# Classification Report Random Forest
st.markdown("**Random Forest - Classification Report**")
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
st.dataframe(pd.DataFrame(rf_report).transpose())

# Classification Report Logistic Regression
st.markdown("**Logistic Regression - Classification Report**")
logreg_report = classification_report(y_test, y_pred_logreg, output_dict=True)
st.dataframe(pd.DataFrame(logreg_report).transpose())

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Depression Detection Project | Built with ❤️ using Streamlit")
