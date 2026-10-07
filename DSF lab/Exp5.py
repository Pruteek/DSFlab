import pandas as pd

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\diabetes.csv"
)

print("Dataset:")
print(data.head())

column = data["Glucose"]

print("\nFrequency:")
print(column.value_counts())

print("\nMean:")
print(column.mean())

print("\nMedian:")
print(column.median())

print("\nMode:")
print(column.mode()[0])

print("\nRange:")
print(column.max() - column.min())

print("\nVariance:")
print(column.var())

print("\nStandard Deviation:")
print(column.std())

print("\nInterquartile Range:")
print(column.quantile(0.75) - column.quantile(0.25))