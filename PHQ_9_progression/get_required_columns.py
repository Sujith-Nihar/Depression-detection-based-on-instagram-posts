import pandas as pd

# Load the Excel file
file_path = "final_analysis/cleaned_file_3.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path, dtype=str)  # Load as string to avoid issues with data types

# Drop the last 19 columns
df = df.iloc[:, :-19]

# Strip whitespace from column names (important if there are hidden spaces)
df.columns = df.columns.str.strip()

# Define the columns to keep
columns_to_keep = [
    'Media Name', 'Profile Name', 'Simple Description', 'Embedded Text', 'Media Caption',
    'Loss of Interest Binary', 'Feeling depressed Binary',
    'Sleeping Disorder Binary', 'Lack of Energy Binary',
    'Eating Disorder Binary', 'Low Self-Esteem Binary',
    'Concentration difficulty Binary', 'Psychomotor changes Binary',
    'Self harm risk Binary'
]


df_filtered = df[df['Profile Name'].str.strip() == 'southernconstellation']

# Select only the specified columns
df_filtered = df_filtered[columns_to_keep]

# List of mental health columns for filtering
mental_health_columns = [
    'Loss of Interest Binary', 'Feeling depressed Binary',
    'Sleeping Disorder Binary', 'Lack of Energy Binary',
    'Eating Disorder Binary', 'Low Self-Esteem Binary',
    'Concentration difficulty Binary', 'Psychomotor changes Binary',
    'Self harm risk Binary'
]

# Ensure correct filtering by stripping spaces, handling NaNs, and checking for "Yes"
df_filtered = df_filtered[df_filtered[mental_health_columns].apply(
    lambda row: row.str.strip().str.lower().fillna("").eq("yes").any(), axis=1
)]

# Save the filtered data
output_filename = "/Users/sujiththota/Downloads/Python/Research/southernconstellation_results/southernconstellation_PHQ_filtered.xlsx"
df_filtered.to_excel(output_filename, index=False)

print(f"Filtered data saved as {output_filename}")
