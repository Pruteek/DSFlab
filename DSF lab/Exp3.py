import pandas as pd

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\diabetes.csv"
)

print("Original Dataset:")
print(data.head())

print("\nMissing Values Before Pre-processing:")
print(data.isnull().sum())

data = data.dropna()

print("\nMissing Values After Removing Rows:")
print(data.isnull().sum())

print("\nDataset After Pre-processing:")
print(data.head())

print("\nShape of Dataset:")
print(data.shape)