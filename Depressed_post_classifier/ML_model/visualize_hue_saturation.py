# --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# --- Load Cleaned Data ---
file_path = '/Users/sujiththota/Downloads/Python/Research/ML_DATA/cleaned_merged_output.xlsx'  # Make sure file is in working directory
df = pd.read_excel(file_path)

# --- Histogram with KDE for Brightness ---
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Brightness Value', hue='Depressed post', kde=True, element='step', stat='density', common_norm=False)
plt.title('Brightness Distribution by Depression Status')
plt.xlabel('Brightness Value')
plt.ylabel('Density')
plt.show()

# --- Violin Plots for Brightness, Saturation, Hue ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Violin Plots of Brightness, Saturation, and Hue by Depression Status', fontsize=16)

sns.violinplot(x='Depressed post', y='Brightness Value', data=df, ax=axs[0], palette={'Yes':'red', 'No':'green'})
axs[0].set_title('Brightness Value')

sns.violinplot(x='Depressed post', y='Saturation Value', data=df, ax=axs[1], palette={'Yes':'red', 'No':'green'})
axs[1].set_title('Saturation Value')

sns.violinplot(x='Depressed post', y='Hue value', data=df, ax=axs[2], palette={'Yes':'red', 'No':'green'})
axs[2].set_title('Hue Value')

for ax in axs:
    ax.set_xlabel('Depressed Post')
    ax.set_ylabel('Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Swarm Plot Over Boxplot for Saturation Value ---
plt.figure(figsize=(8,6))
sns.boxplot(x='Depressed post', y='Saturation Value', data=df, palette={'Yes':'red', 'No':'green'})
sns.swarmplot(x='Depressed post', y='Saturation Value', data=df, color='.25')
plt.title('Saturation Value Distribution with Swarmplot')
plt.xlabel('Depressed Post')
plt.ylabel('Saturation Value')
plt.show()

# --- Facet Grid for Brightness Value ---
g = sns.FacetGrid(df, col="Depressed post", height=5)
g.map(sns.histplot, "Brightness Value", kde=True)
plt.show()

# --- Correlation Heatmap ---
df_corr = df.copy()
df_corr['Depressed post'] = df_corr['Depressed post'].map({'Yes': 1, 'No': 0})

plt.figure(figsize=(10,8))
sns.heatmap(df_corr[['Brightness Value', 'Saturation Value', 'Hue value', 'Depressed post']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# --- Statistical Tests ---
# Brightness
brightness_yes = df[df['Depressed post'] == 'Yes']['Brightness Value']
brightness_no = df[df['Depressed post'] == 'No']['Brightness Value']
t_stat_brightness, p_value_brightness = ttest_ind(brightness_yes, brightness_no, equal_var=False)

# Saturation
saturation_yes = df[df['Depressed post'] == 'Yes']['Saturation Value']
saturation_no = df[df['Depressed post'] == 'No']['Saturation Value']
t_stat_saturation, p_value_saturation = ttest_ind(saturation_yes, saturation_no, equal_var=False)

# Hue
hue_yes = df[df['Depressed post'] == 'Yes']['Hue value']
hue_no = df[df['Depressed post'] == 'No']['Hue value']
t_stat_hue, p_value_hue = ttest_ind(hue_yes, hue_no, equal_var=False)

print(f"Brightness Value - T-Statistic: {t_stat_brightness:.4f}, P-Value: {p_value_brightness:.4f}")
print(f"Saturation Value - T-Statistic: {t_stat_saturation:.4f}, P-Value: {p_value_saturation:.4f}")
print(f"Hue Value - T-Statistic: {t_stat_hue:.4f}, P-Value: {p_value_hue:.4f}")
