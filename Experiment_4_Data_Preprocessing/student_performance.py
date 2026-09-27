import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

df = pd.read_csv("Student_Performance_Dataset.csv")

print("Student Performance Dataset:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].median())
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].median())
df['Assignment_Score'] = df['Assignment_Score'].fillna(df['Assignment_Score'].median())
df['Internal_Marks'] = df['Internal_Marks'].fillna(df['Internal_Marks'].median())
df['Final_Marks'] = df['Final_Marks'].fillna(df['Final_Marks'].median())

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

print("\nDataset After Gender Encoding:")
print(df.head())

for column in ['Study_Hours', 'Final_Marks']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print("\n", column, "Outliers:")
    print(outliers)

    df[column] = df[column].clip(lower, upper)

plt.figure(figsize=(8, 5))
plt.boxplot([df['Study_Hours'], df['Final_Marks']])
plt.xticks([1, 2], ['Study_Hours', 'Final_Marks'])
plt.title("Box Plot of Study Hours and Final Marks")
plt.ylabel("Values")
plt.show()

columns = [
    'Age',
    'Attendance',
    'Study_Hours',
    'Assignment_Score',
    'Internal_Marks',
    'Final_Marks'
]

scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns] = scaler.fit_transform(df_normalized[columns])

print("\nMin-Max Normalized Data:")
print(df_normalized[columns].head())

standard_scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[columns] = standard_scaler.fit_transform(df_standardized[columns])

print("\nStandardized Data:")
print(df_standardized[columns].head())

print("\nComparison of Normalized and Standardized Data:")
print("Normalized:")
print(df_normalized[columns].head())

print("\nStandardized:")
print(df_standardized[columns].head())

df.to_csv("student_preprocessed.csv", index=False)

print("\nDataset saved as student_preprocessed.csv")