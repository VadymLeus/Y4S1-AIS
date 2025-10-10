import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Функція для візуалізації меж класифікатора
def plot_classifier(classifier, X, y, title):
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    
    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Завантаження вхідних даних
    input_file = 'data_imbalance.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Поділ вхідних даних на два класи на підставі міток
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])

    # Візуалізація вхідних даних
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
                edgecolors='black', linewidth=1, marker='x', label='Клас 0 (Більшість)')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o', label='Клас 1 (Меншість)')
    plt.title('Вхідні дані з дисбалансом класів')
    plt.legend()
    plt.show()

    # Розбиття даних на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    # Класифікатор на основі гранично випадкових лісів
    # Визначаємо параметри. Перевіряємо, чи передано аргумент 'balance' з командного рядка
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if len(sys.argv) > 1 and sys.argv[1] == 'balance':
        params['class_weight'] = 'balanced'
        title = 'Класифікатор з урахуванням дисбалансу'
        print("Виконується класифікація з параметром class_weight='balanced'.")
    else:
        title = 'Класифікатор без урахування дисбалансу'
        print("Виконується класифікація без урахування дисбалансу класів.")

    # Створення, навчання та візуалізація класифікатора
    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)
    plot_classifier(classifier, X_test, y_test, f'{title} (тестові дані)')

    # Обчислення показників ефективності класифікатора
    y_test_pred = classifier.predict(X_test)
    print(f"\nЗвіт про якість класифікації для '{title}':\n")
    print(classification_report(y_test, y_test_pred))