import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt' 

try:
    data = np.loadtxt(input_file, delimiter=',')
except FileNotFoundError:
    print(f"Помилка: Файл '{input_file}' не знайдено.")
    exit()

X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()

linear_regressor.fit(X_train, y_train)

y_test_pred_linear = linear_regressor.predict(X_test)
print("====================================")
print("Linear Regressor performance:")
print("Mean absolute error =", 
      round(sm.mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Mean squared error =", 
      round(sm.mean_squared_error(y_test, y_test_pred_linear), 2))
print("Median absolute error =", 
      round(sm.median_absolute_error(y_test, y_test_pred_linear), 2))
print("Explained variance score =", 
      round(sm.explained_variance_score(y_test, y_test_pred_linear), 2))
print("R2 score =", 
      round(sm.r2_score(y_test, y_test_pred_linear), 2))
print("====================================")

polynomial = PolynomialFeatures(degree=10)

X_train_transformed = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()

poly_linear_model.fit(X_train_transformed, y_train)

X_test_transformed = polynomial.transform(X_test)
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

print("\n====================================")
print("Polynomial Regressor (degree=10) performance:")
print("Mean absolute error =", 
      round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", 
      round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("R2 score =", 
      round(sm.r2_score(y_test, y_test_pred_poly), 2))
print("====================================")

datapoint = np.array([[7.75, 6.35, 5.56]])

pred_linear = linear_regressor.predict(datapoint)[0]

poly_datapoint = polynomial.transform(datapoint)
pred_poly = poly_linear_model.predict(poly_datapoint)[0]

print(f"\nTarget datapoint: [7.66, 6.29, 5.66] -> ~41.35")
print(f"Prediction for input {datapoint[0]} (Expected ~41.35):")
print(f"Linear regression prediction: {round(pred_linear, 2)}")
print(f"Polynomial regression prediction: {round(pred_poly, 2)}")

