import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("Online Retail.xlsx")

print("Online Retail Sales Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

before = df.describe(include='all')

df = df.drop_duplicates()

print("\nDuplicates After Removal:")
print(df.duplicated().sum())

df['CustomerID'] = df['CustomerID'].fillna(
    df['CustomerID'].median()
)

df['InvoiceDate'] = pd.to_datetime(
    df['InvoiceDate'],
    errors='coerce'
)

print("\nMissing Values After Imputation:")
print(df[['CustomerID', 'InvoiceDate']].isnull().sum())

df = pd.get_dummies(
    df,
    columns=['Country'],
    dtype=int
)

print("\nDataset After Country Encoding:")
print(df.head())

for column in ['Quantity', 'UnitPrice']:
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

scaler = MinMaxScaler()

df[['Quantity', 'UnitPrice']] = scaler.fit_transform(
    df[['Quantity', 'UnitPrice']]
)

print("\nNormalized Quantity and UnitPrice:")
print(df[['Quantity', 'UnitPrice']].head())

after = df.describe(include='all')

print("\nSummary Statistics Before Preprocessing:")
print(before)

print("\nSummary Statistics After Preprocessing:")
print(after)

df.to_csv(
    "retail_sales_cleaned.csv",
    index=False
)

print("\nDataset saved as retail_sales_cleaned.csv")