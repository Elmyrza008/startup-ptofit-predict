import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('50_Startups.csv')

data.columns = data.columns.str.strip().str.lower()

data = pd.get_dummies(data, columns=["state"], drop_first=True)

X = data.drop("profit", axis=1)
y = data["profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 score: {r2:.4f}")


joblib.dump(model, 'startup_model.pkl')
print("Модель успешно сохранена в файл startup_model.pkl")

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.7)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)

plt.xlabel("Real Profit")
plt.ylabel("Predicted Profit")
plt.title("Linear Regressiosn: Real vs Predicted")
plt.grid(True)

plt.show()
