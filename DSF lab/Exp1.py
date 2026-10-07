import pandas as pd

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\annual-enterprise-survey-2025-financial-year-provisional-size-bands.csv"
)

print(data.head())
print(data.shape)
print(data.info())
print(data.describe())