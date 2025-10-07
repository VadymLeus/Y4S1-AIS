import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

input_file = 'data_regr_4.txt' 

try:
    data = np.loadtxt(input_file, delimiter=',')
except FileNotFoundError:
    print(f"Помилка: Файл '{input_file}' не знайдено. Переконайтеся, що він є у поточному каталозі.")
    exit()

X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]

regressor = linear_model.LinearRegression()

regressor.fit(X_train, y_train)

y_test_pred = regressor.predict(X_test)

plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='green', label='Справжні тестові дані')

plt.plot(X_test, y_test_pred, color='black', linewidth=4, label='Лінія регресії')

plt.title(f'Лінійна регресія однієї змінної (Варіант 4: {input_file})')
plt.xlabel('Незалежна змінна X')
plt.ylabel('Залежна змінна Y')

plt.legend()
plt.grid(True)
plt.show()

print("Linear regressor performance (Variant 4):")
print("Mean absolute error (MAE) =", 
      round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error (MSE) =", 
      round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", 
      round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", 
      round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

output_model_file = 'model_4.pkl'

with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
print(f"\nМодель регресора успішно збережено у файл '{output_model_file}'.")

with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

y_test_pred_new = regressor_model.predict(X_test)

print("New mean absolute error (MAE) from loaded model =", 
      round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))