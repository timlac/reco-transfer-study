import pandas as pd
import os

path = "/home/tim/Downloads/new_export/"

# Get all CSV files in the directory
csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

# Read and merge all CSV files
merged_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# Show the merged DataFrame
print(merged_df)

merged_df.to_csv("/home/tim/Downloads/new_export/merged_surveys.csv", index=False)