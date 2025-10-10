import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    parameter_grid = [
        {'n_estimators': [25, 50, 100, 250], 'max_depth': [2, 4, 7, 12, 16]},
    ]

    metrics = ['precision_weighted', 'recall_weighted']
    for metric in metrics:
        print("\n############################################################")
        print(f"Пошук оптимальних параметрів для метрики: '{metric}'")
        print("############################################################")

        classifier = GridSearchCV(
            ExtraTreesClassifier(random_state=0),
            parameter_grid,
            cv=5,
            scoring=metric
        )

        classifier.fit(X_train, y_train)
        print("\nОцінки для кожної комбінації параметрів:")
        for mean_score, params in zip(classifier.cv_results_['mean_test_score'], classifier.cv_results_['params']):
            print(f"  Оцінка: {mean_score:.4f} для параметрів: {params}")

        print(f"\nНайкращі параметри для метрики '{metric}':")
        print(classifier.best_params_)
        
        print(f"\nНайкраща оцінка на перехресній перевірці ({metric}): {classifier.best_score_:.4f}")
        y_pred = classifier.predict(X_test)
        
        print("\nЗвіт класифікації для тестового набору даних:")
        print(classification_report(y_test, y_pred))