import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

m = 100
np.random.seed(42)
X_flat = np.linspace(-3, 3, m)
y = 3 + np.sin(X_flat) + np.random.uniform(-0.5, 0.5, m)
X = X_flat.reshape(-1, 1) 
def plot_learning_curves(model, X, y, title):
    """
    Побудова кривих навчання: помилка тренувального та валідаційного наборів 
    як функція від розміру тренувального набору.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_errors, val_errors = [], []
    for m_size in range(1, len(X_train)):
        model.fit(X_train[:m_size], y_train[:m_size])
        y_train_predict = model.predict(X_train[:m_size])
        y_val_predict = model.predict(X_val)
        train_errors.append(np.sqrt(mean_squared_error(y_train_predict, y_train[:m_size])))
        val_errors.append(np.sqrt(mean_squared_error(y_val_predict, y_val)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, "r-+", linewidth=2, label="Навчальний набір (train)")
    plt.plot(val_errors, "b-", linewidth=3, label="Перевірочний набір (val)")
    plt.legend(loc="upper right")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.ylim(0, 1.5)
    plt.grid(True)
    plt.show()

print("Криві навчання для чисто лінійної регресії (Недонавчання):")
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання для лінійної моделі")

print("\nКриві навчання для поліноміальної регресії 10-го ступеня (Перенавчання):")
poly_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg_10, X, y, "Криві навчання для поліноміальної моделі (ступінь 10)")

print("\nКриві навчання для поліноміальної регресії 2-го ступеня (Оптимальна складність):")
poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg_2, X, y, "Криві навчання для поліноміальної моделі (ступінь 2)")