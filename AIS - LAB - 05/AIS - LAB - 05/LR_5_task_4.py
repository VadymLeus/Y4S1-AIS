import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == '__main__':

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                              'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    X, y = shuffle(data, target, random_state=7)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    regressor = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=4),
        n_estimators=400, random_state=7)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("\nADABOOST REGRESSOR")
    print(f"Mean squared error = {round(mse, 2)}")
    print(f"Explained variance score = {round(evs, 2)}")

    feature_importances = regressor.feature_importances_

    feature_importances = 100.0 * (feature_importances / feature_importances.max())

    index_sorted = np.flipud(np.argsort(feature_importances))
    
    pos = np.arange(index_sorted.shape[0]) + 0.5

    plt.figure(figsize=(12, 7))
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted], rotation=45, ha='right')
    plt.ylabel('Відносна важливість (%)')
    plt.title('Важливість ознак для прогнозування цін на нерухомість')
    plt.tight_layout()
    plt.show()
