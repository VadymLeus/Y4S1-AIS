import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import warnings

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

RANDOM_STATE = 1
TEST_SIZE = 0.20
N_SPLITS = 10

def step_1_data_loading():
    iris_dataset = load_iris()

    print("="*60)
    print("КРОК 1. ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ")
    print("="*60)

    print("Ключі iris_dataset:", iris_dataset.keys())
    print("-" * 30)
    print("Назви відповідей:", iris_dataset['target_names'])
    print("Назва ознак:", iris_dataset['feature_names'])
    print("-" * 30)
    print("Форма масиву data:", iris_dataset['data'].shape)
    print("Перші 5 прикладів:\n", iris_dataset['data'][:5])
    print("Відповіді:", iris_dataset['target'])
    print("-" * 30)

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)

    print(f"Форма датасету: {dataset.shape}")
    print("\nПерші 20 рядків:")
    print(dataset.head(20))
    print("\nСтатистичне зведення:")
    print(dataset.describe())
    print("\nРозподіл за класами:")
    print(dataset.groupby('class').size())
    
    return dataset

def step_2_visualization(dataset):
    print("\n" + "="*60)
    print("КРОК 2. ВІЗУАЛІЗАЦІЯ ДАНИХ")
    print("="*60)

    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.suptitle('Діаграма розмаху атрибутів')
    plt.show()

    dataset.hist()
    plt.suptitle('Гістограма розподілу атрибутів')
    plt.show()

    scatter_matrix(dataset, figsize=(12, 12))
    plt.suptitle('Матриця діаграм розсіювання')
    plt.show()

def step_3_split_data(dataset):
    print("\n" + "="*60)
    print("КРОК 3. СТВОРЕННЯ НАВЧАЛЬНОГО ТА ТЕСТОВОГО НАБОРІВ")
    print("="*60)

    array = dataset.values
    X = array[:, 0:4].astype(float)
    y = array[:, 4]

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"X_train.shape: {X_train.shape}")
    print(f"X_validation.shape: {X_validation.shape}")
    
    return X_train, X_validation, Y_train, Y_validation

def step_4_compare_models(X_train, Y_train):
    print("\n" + "="*60)
    print("КРОК 4. ПОРІВНЯННЯ МОДЕЛЕЙ")
    print("="*60)

    models = [
        ('LR', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=RANDOM_STATE)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ('NB', GaussianNB()),
        ('SVM', SVC(gamma='auto', random_state=RANDOM_STATE))
    ]

    results = []
    names = []
    print("Результати 10-кратної крос-валідації (Accuracy):")

    for name, model in models:
        kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %.4f (%.4f)' % (name, cv_results.mean(), cv_results.std()))

    plt.boxplot(results, labels=names)
    plt.title('Порівняння алгоритмів (Accuracy)')
    plt.show()
    
    best_name = names[np.argmax([r.mean() for r in results])]
    return best_name

def step_6_7_evaluate_best(X_train, X_validation, Y_train, Y_validation, best_model_name):
    print("\n" + "="*60)
    print(f"КРОК 6 & 7. ОЦІНКА МОДЕЛІ ({best_model_name})")
    print("="*60)
    
    if best_model_name == 'SVM':
        best_model = SVC(gamma='auto', random_state=RANDOM_STATE)
    elif best_model_name == 'LDA':
        best_model = LinearDiscriminantAnalysis()
    elif best_model_name == 'KNN':
        best_model = KNeighborsClassifier()
    else:
        best_model = SVC(gamma='auto', random_state=RANDOM_STATE)
        best_model_name = 'SVM'

    best_model.fit(X_train, Y_train)
    predictions = best_model.predict(X_validation)

    print(f"🔹 Оцінка {best_model_name} на контрольній вибірці 🔹")
    print(f"Точність: {accuracy_score(Y_validation, predictions):.4f}")

    print("\nМатриця помилок:")
    print(confusion_matrix(Y_validation, predictions))

    print("\nЗвіт про класифікацію:")
    print(classification_report(Y_validation, predictions))
    
    return best_model

def step_8_predict_new(best_model):
    print("\n" + "="*60)
    print("КРОК 8. ПРОГНОЗ ДЛЯ НОВИХ ДАНИХ")
    print("="*60)

    X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
    prediction = best_model.predict(X_new)
    predicted_label = prediction[0]

    print(f"Форма масиву X_new: {X_new.shape}")
    print(f"Спрогнозована мітка: {predicted_label}")
    print("="*60)

if __name__ == "__main__":
    data_frame = step_1_data_loading()
    step_2_visualization(data_frame)
    X_train, X_validation, Y_train, Y_validation = step_3_split_data(data_frame)
    best_model_name = step_4_compare_models(X_train, Y_train)
    best_model = step_6_7_evaluate_best(X_train, X_validation, Y_train, Y_validation, best_model_name)
    step_8_predict_new(best_model)