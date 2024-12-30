import pandas as pd

# File paths
og_file = r'E:\Mars_Dust_Storm\MDAD\data\MDAD.csv'
pp_file = r'E:\Mars_Dust_Storm\MDAD\data\MDAD_refined_cleaned.csv'

# Load CSV files
og_df = pd.read_csv(og_file)
pp_df = pd.read_csv(pp_file)

# Display structure of the original CSV
print("== OG CSV Details ==")
print("Cols:", og_df.columns.tolist())
print("\nTop 5 rows from OG CSV:")
print(og_df.head())

# Display structure of the pre-processed CSV
print("\n== PP CSV Details ==")
print("Cols:", pp_df.columns.tolist())
print("\nTop 5 rows from PP CSV:")
print(pp_df.head())
