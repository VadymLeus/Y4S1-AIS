import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    input_file = 'traffic_data.txt'
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            items = line.strip().split(',')
            data.append(items)

    data = np.array(data)

    label_encoders = []
    X_encoded = np.empty(data.shape)

    for i, item in enumerate(data[0]):
        if item.isdigit():
            X_encoded[:, i] = data[:, i]
        else:
            encoder = preprocessing.LabelEncoder()
            label_encoders.append(encoder)
            X_encoded[:, i] = encoder.fit_transform(data[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    regressor = ExtraTreesRegressor(**params)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    print(f"Mean absolute error: {round(mean_absolute_error(y_test, y_pred), 2)}")

    test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
    test_datapoint_encoded = [-1] * len(test_datapoint)
    count = 0

    print("\nOriginal test data point:", test_datapoint)

    for i, item in enumerate(test_datapoint):
        if item.isdigit():
            test_datapoint_encoded[i] = int(test_datapoint[i])
        else:
            encoded_value = label_encoders[count].transform([test_datapoint[i]])[0]
            test_datapoint_encoded[i] = int(encoded_value)
            count += 1
    
    print("Encoded test data point:", test_datapoint_encoded)

    predicted_traffic = regressor.predict([test_datapoint_encoded])
    print(f"\nPredicted traffic for the test data point: {int(predicted_traffic[0])}")



