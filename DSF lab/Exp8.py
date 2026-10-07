import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\diabetes.csv"
)

x = data[["Glucose"]]
y = data["BMI"]

correlation = data["Glucose"].corr(data["BMI"])

print("Correlation Coefficient:", correlation)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R-squared:", r2_score(y, y_pred))

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Correlation and Regression Analysis")
plt.show()