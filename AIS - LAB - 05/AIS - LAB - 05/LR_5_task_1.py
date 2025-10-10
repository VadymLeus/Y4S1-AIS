import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def plot_classifier(classifier, X, y):
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)

    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Створення класифікаторів на основі лісів')
    parser.add_argument('--classifier-type', dest='classifier_type', required=True,
                        choices=['rf', 'erf'], help='Тип класифікатора: "rf" для випадкового лісу або "erf" для гранично випадкового лісу')
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    class_0 = np.array(X[y==0])
    class_1 = np.array(X[y==1])
    class_2 = np.array(X[y==2])

    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='s', label='Клас 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o', label='Клас 1')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='^', label='Клас 2')
    plt.title('Вхідні дані')
    plt.legend()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
        title = 'Випадковий ліс'
    else:
        classifier = ExtraTreesClassifier(**params)
        title = 'Гранично випадковий ліс'

    classifier.fit(X_train, y_train)

    plt.figure()
    plot_classifier(classifier, X, y)
    plt.title(title)
    plt.show()

    y_test_pred = classifier.predict(X_test)
    print(f"\nЗвіт про якість класифікації для {title}:\n")
    print(classification_report(y_test, y_test_pred))

    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2.5]])
    print("Параметри довірливості для нових точок даних:")
    
    for datapoint in test_datapoints:
        plt.figure()
        
        plot_classifier(classifier, X, y) 
        plt.scatter(datapoint[0], datapoint[1], s=200, facecolors='none', edgecolors='red', linewidth=3, marker='x')


        plt.title(f'{title}: Тестова точка [{datapoint[0]}, {datapoint[1]}]')
        plt.show()

        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = classifier.predict([datapoint])[0]
        
        print(f"\nТочка даних: {datapoint}")
        print(f"  Ймовірність належності до класу 0: {round(probabilities[0]*100, 2)}%")
        print(f"  Ймовірність належності до класу 1: {round(probabilities[1]*100, 2)}%")
        print(f"  Ймовірність належності до класу 2: {round(probabilities[2]*100, 2)}%")
        print(f"  Передбачений клас: {int(predicted_class)}")