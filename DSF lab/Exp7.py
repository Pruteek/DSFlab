import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\diabetes.csv"
)

print("Statistical Measures:")
print(data.describe())

print("\nCorrelation Matrix:")
print(data.corr())

plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title("Box Plot")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Plot")
plt.show()