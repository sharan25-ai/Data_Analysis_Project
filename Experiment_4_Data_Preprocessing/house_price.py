import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("Housing.csv")

print("House Price Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nSummary Statistics Before Preprocessing:")
print(df.describe())

for column in df.select_dtypes(include='number').columns:
    df[column] = df[column].fillna(df[column].median())

for column in df.select_dtypes(include='object').columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

df = df.drop_duplicates()

print("\nDuplicates After Removal:")
print(df.duplicated().sum())

le = LabelEncoder()
df['airconditioning'] = le.fit_transform(df['airconditioning'])

df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    dtype=int
)

print("\nDataset After Encoding:")
print(df.head())

for column in ['area', 'price']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print("\n", column, "Outliers:", len(outliers))

    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]

print("\nDataset Shape After Outlier Removal:")
print(df.shape)

columns = [
    'area',
    'bedrooms',
    'bathrooms',
    'price'
]

scaler = MinMaxScaler()
df[columns] = scaler.fit_transform(df[columns])

print("\nMin-Max Normalized Data:")
print(df[columns].head())

print("\nSummary Statistics After Preprocessing:")
print(df.describe())

df.to_csv("houseprice_cleaned.csv", index=False)

print("\nDataset saved as houseprice_cleaned.csv")