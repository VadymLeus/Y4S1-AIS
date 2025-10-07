import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target 

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.5, random_state=0
)
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Лінійний регресор на наборі даних про діабет:")

print("\nКоефіцієнти (Coefficients):")
print(np.round(regr.coef_, 2)) 

print(f"\nПеретин (Intercept): {round(regr.intercept_, 2)}")

print("\nМетрики оцінки якості:")

print(f"Коефіцієнт кореляції R2 (R2 score): {round(r2_score(ytest, ypred), 2)}")

print(f"Середня абсолютна помилка (MAE): {round(mean_absolute_error(ytest, ypred), 2)}")

print(f"Середньоквадратична помилка (MSE): {round(mean_squared_error(ytest, ypred), 2)}")

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), color='green', alpha=0.6, label='Прогнози моделі')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ідеальний прогноз (y=x)')

ax.set_xlabel('Виміряно (Справжня прогресія захворювання)')
ax.set_ylabel('Передбачено (Прогнозована прогресія захворювання)')
ax.set_title('Багатовимірна лінійна регресія: Діабет')
ax.legend()
ax.grid(True)
plt.show()