import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("Healthcare_Patient_Dataset.csv")

print("Healthcare Patient Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

numerical_columns = df.select_dtypes(include='number').columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

le = LabelEncoder()

if 'Gender' in df.columns:
    df['Gender'] = le.fit_transform(df['Gender'])

if 'Diagnosis' in df.columns:
    df['Diagnosis'] = le.fit_transform(df['Diagnosis'])

print("\nDataset After Label Encoding:")
print(df.head())

numeric_columns = df.select_dtypes(include='number').columns

for column in ['Blood_Pressure', 'Blood_Sugar']:
    if column in df.columns:
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

        plt.figure(figsize=(6, 4))
        plt.boxplot(df[column])
        plt.title("Box Plot of " + column)
        plt.ylabel(column)
        plt.show()

columns = [
    column for column in
    ['Age', 'Height', 'Weight', 'Blood_Pressure', 'Blood_Sugar']
    if column in df.columns
]

scaler = MinMaxScaler()

df[columns] = scaler.fit_transform(df[columns])

print("\nDataset After Min-Max Scaling:")
print(df.head())

correlation_matrix = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix of Numerical Variables")
plt.show()

df.to_csv("healthcare_processed.csv", index=False)

print("\nProcessed dataset saved successfully as healthcare_processed.csv")