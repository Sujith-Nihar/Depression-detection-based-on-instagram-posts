import pandas as pd
import os

# Path where your Excel files are stored
folder_path = '/Users/sujiththota/Downloads/Python/Research/ML_DATA/'  # <-- change this

# List to collect each DataFrame
dfs = []

# Loop through all Excel files in the folder
for file in os.listdir(folder_path):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)  # Read each file
        dfs.append(df)  # Add to the list

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe to a new Excel file
merged_df.to_excel('/Users/sujiththota/Downloads/Python/Research/ML_DATA/merged_output.xlsx', index=False)

print("✅ Successfully merged and saved as 'merged_output.xlsx'")
