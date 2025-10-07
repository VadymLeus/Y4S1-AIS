import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

m = 100
np.random.seed(42)
X_flat = np.linspace(-3, 3, m)
y = 3 + np.sin(X_flat) + np.random.uniform(-0.5, 0.5, m)

X = X_flat.reshape(-1, 1)

lin_reg_simple = LinearRegression()
lin_reg_simple.fit(X, y)
y_pred_simple = lin_reg_simple.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("====================================")
print("Перетворення ознак для поліноміальної регресії:")
print(f"Перший елемент X (первинна ознака): X[0] = {X[0][0]:.4f}")
print(f"Перший елемент X_poly (перетворені ознаки): X_poly[0] = {X_poly[0]}")

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

intercept = lin_reg_poly.intercept_
coef_x1 = lin_reg_poly.coef_[0]
coef_x2 = lin_reg_poly.coef_[1]

y_pred_poly = lin_reg_poly.predict(X_poly)

print("\nКоефіцієнти Поліноміальної Регресії (ступінь 2):")
print(f"Перетин (Intercept): {intercept:.3f}")
print(f"Коефіцієнт для X (coef_x1): {coef_x1:.3f}")
print(f"Коефіцієнт для X^2 (coef_x2): {coef_x2:.3f}")

print("\nОцінка якості:")
print(f"MAE (Поліноміальна): {mean_squared_error(y, y_pred_poly):.3f}")
print(f"R2 score (Поліноміальна): {r2_score(y, y_pred_poly):.3f}")
print("====================================")

plt.figure(figsize=(10, 6))

plt.scatter(X, y, color='blue', alpha=0.6, label='Згенеровані дані')

plt.plot(X, y_pred_simple, color='orange', linestyle='--', linewidth=2, label='Лінійна регресія (y = {:.2f}x + {:.2f})'.format(lin_reg_simple.coef_[0], lin_reg_simple.intercept_))

X_plot = np.sort(X_flat).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot_poly = lin_reg_poly.predict(X_plot_poly)

plt.plot(X_plot, y_plot_poly, color='red', linewidth=3, label=f'Поліноміальна регресія (ступінь 2)')

plt.title('Порівняння Лінійної та Поліноміальної Регресії (Варіант 9)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()