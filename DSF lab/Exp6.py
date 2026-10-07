import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\diabetes.csv"
)

glucose = data["Glucose"]

print("Mean Glucose:", glucose.mean())

t_stat, p_value = ttest_1samp(glucose, 120)

print("\nHypothesis Testing:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

mean = glucose.mean()
std = glucose.std()
n = len(glucose)

margin = 1.96 * (std / np.sqrt(n))

print("\nParameter Estimation:")
print("Mean:", mean)
print("95% Confidence Interval:", mean - margin, "to", mean + margin)

x = data.iloc[0][["Glucose", "BloodPressure", "BMI"]].values.astype(float)
y = data.iloc[1][["Glucose", "BloodPressure", "BMI"]].values.astype(float)

euclidean_distance = np.linalg.norm(x - y)

cosine_sim = cosine_similarity([x], [y])[0][0]

print("\nSimilarity and Dissimilarity:")
print("Euclidean Distance:", euclidean_distance)
print("Cosine Similarity:", cosine_sim)