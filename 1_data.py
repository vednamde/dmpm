import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, normaltest
from sklearn.model_selection import train_test_split

# 1. Read the CSV file
df = pd.read_csv("Titanic.csv")
print("Dataset loaded.\n")

# 2. Identify variables and missing values
print("\nMissing values:\n", df.isnull().sum())

# 3. Impute missing values
# Let's fill 'Age' with median, 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Verify imputation
print("\nMissing values after imputation:\n", df.isnull().sum())

# 4. Compute Kurtosis and Skewness
numerical_cols = df.select_dtypes(include=np.number).columns
print("\nKurtosis and Skewness:")
for col in numerical_cols:
    print(f"{col}: Kurtosis = {kurtosis(df[col], nan_policy='omit'):.2f}, Skewness = {skew(df[col], nan_policy='omit'):.2f}")

# 5. Summary statistics
print("\nSummary statistics:\n", df[numerical_cols].describe())

# 6. Plot distributions
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# 7. Natural log transformation
df_log = df[numerical_cols].apply(lambda x: np.log1p(x))  # log1p to avoid log(0)
print("\nLog-transformed data sample:\n", df_log.head())

# 8. Check normality
print("\nNormality Test (D’Agostino and Pearson):")
for col in df_log.columns:
    stat, p = normaltest(df_log[col].dropna())
    print(f"{col}: p-value = {p:.4f} {'(Normal)' if p > 0.05 else '(Not Normal)'}")

# 9. Correlation matrix and heatmap
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 10. Partition into 70-15-15 (Train-Validation-Test)
train_data, temp_data = train_test_split(df, test_size=0.30, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"\nDataset sizes:\nTrain: {len(train_data)}\nValidation: {len(val_data)}\nTest: {len(test_data)}")
