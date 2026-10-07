import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(
    r"C:\Users\prati\OneDrive\Desktop\DSF lab\diabetes.csv"
)

X = data[["Glucose"]]
y = data["BMI"]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R-squared:", r2_score(y, y_pred))

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Simple Linear Regression")
plt.show()