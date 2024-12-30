import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

data_df = pd.read_csv(r'E:\Mars_Dust_Storm\MDAD\data\MDAD.csv')
print("Cols b4 rename:", data_df.columns.tolist())

data_df.rename(columns={
    'Ls': 'Solar_Long',
    'Centroid longitude': 'Centroid_Long',
    'Centroid latitude': 'Centroid_Lat',
    'Area (square km)': 'Area_km2',
    'Confidence interval': 'Conf_Level',
    'Maximum latitude': 'Max_Lat',
    'Minimum latitude': 'Min_Lat'
}, inplace=True)
print("Cols after rename:", data_df.columns.tolist())

data_df['Seq_ID'] = data_df['Sequence ID'].fillna('Unknown')
data_df['Missing_data'] = data_df['Missing data'].str.lower()

num_cols = ['Solar_Long', 'Centroid_Long', 'Centroid_Lat', 'Area_km2', 'Max_Lat', 'Min_Lat', 'Conf_Level']
for col in num_cols:
    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

critical_cols = ['Solar_Long', 'Centroid_Long', 'Centroid_Lat']
data_df.dropna(subset=critical_cols, inplace=True)

data_df['Lat_Range'] = data_df['Max_Lat'] - data_df['Min_Lat']
data_df['Norm_Area'] = (data_df['Area_km2'] - data_df['Area_km2'].mean()) / data_df['Area_km2'].std()
data_df['Norm_Lat_Range'] = (data_df['Lat_Range'] - data_df['Lat_Range'].mean()) / data_df['Lat_Range'].std()

mission_map = {name: idx for idx, name in enumerate(data_df['Mission subphase'].unique())}
data_df['Mission_Phase_Enc'] = data_df['Mission subphase'].map(mission_map)

data_df['Long_x_Lat'] = data_df['Centroid_Long'] * data_df['Centroid_Lat']
data_df['Solar_x_Area'] = data_df['Solar_Long'] * data_df['Area_km2']

imputer = SimpleImputer(strategy='mean')
data_df[num_cols] = imputer.fit_transform(data_df[num_cols])

scaler = RobustScaler()
data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

pca = PCA(n_components=5)
pca_feats = pca.fit_transform(data_df[num_cols])
pca_cols = [f'PCA_{i+1}' for i in range(pca_feats.shape[1])]
pca_df = pd.DataFrame(pca_feats, columns=pca_cols)
data_df = pd.concat([data_df, pca_df], axis=1)

q1_area = data_df['Area_km2'].quantile(0.25)
q3_area = data_df['Area_km2'].quantile(0.75)
iqr_area = q3_area - q1_area
low_area = q1_area - 1.5 * iqr_area
high_area = q3_area + 1.5 * iqr_area
cleaned_df = data_df[(data_df['Area_km2'] >= low_area) & (data_df['Area_km2'] <= high_area)]

q1_lat = data_df['Lat_Range'].quantile(0.25)
q3_lat = data_df['Lat_Range'].quantile(0.75)
iqr_lat = q3_lat - q1_lat
low_lat = q1_lat - 1.5 * iqr_lat
high_lat = q3_lat + 1.5 * iqr_lat
cleaned_df = cleaned_df[(cleaned_df['Lat_Range'] >= low_lat) & (cleaned_df['Lat_Range'] <= high_lat)]

print("Cleaned shape:", cleaned_df.shape)
summary = cleaned_df.describe(include='all')
print(summary)

cleaned_path = r'E:\Mars_Dust_Storm\MDAD\data\MDAD_refined_cleaned.csv'
cleaned_df.to_csv(cleaned_path, index=False)
print(f"Cleaned data saved to: {cleaned_path}")
