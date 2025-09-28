import numpy as np
import pandas as pd
import re
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
INPUT_FILE = 'income_data.txt'
RANDOM_STATE = 5
TEST_SIZE = 0.2
N_SPLITS = 5 

def clean_data(data):
    """Очистка даних: видалення '?', пробілів, переведення у нижній регістр."""
    data = data.replace(r'^\s*\?+\s*$', np.nan, regex=True).dropna()

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str).str.strip().str.lower()
        data[col] = data[col].apply(lambda x: re.sub(r'[^a-z0-9<=->]', '', x))

    return data

def encode_categorical_data(data):
    """Кодування категоріальних змінних за допомогою LabelEncoder."""
    label_encoders = {}
    data_encoded = data.copy()

    for column in data.columns:
        if data[column].dtype == 'object':
            le = preprocessing.LabelEncoder()
            data_encoded[column] = le.fit_transform(data_encoded[column])
            label_encoders[column] = le
    
    for col in data_encoded.columns:
        if data_encoded[col].dtype != 'object':
             data_encoded[col] = data_encoded[col].astype(int)

    return data_encoded, label_encoders

def scale_data(X_train, X_validation):
    """Масштабування числових ознак за допомогою StandardScaler."""
    
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_validation_scaled = X_validation.copy()
    
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_validation_scaled[numeric_features] = scaler.transform(X_validation[numeric_features])
    
    return X_train_scaled, X_validation_scaled, scaler

def main():
    
    print("="*80)
    print("ЗАВДАННЯ 2.4: ПОРІВНЯННЯ КЛАСИФІКАТОРІВ НАБОРУ ДАНИХ INCOME_DATA")
    print("="*80)

    try:
        data = pd.read_csv(INPUT_FILE, header=None, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values=['?'])
        data = clean_data(data)
        data_encoded, label_encoders = encode_categorical_data(data)

        X = data_encoded.drop('income', axis=1)
        y = data_encoded['income']
        
        X_train, X_validation, Y_train, Y_validation = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        print(f"Розмір навчальної вибірки: {X_train.shape}")
        print(f"Розмір контрольної вибірки: {X_validation.shape}\n")

        X_train_scaled, X_validation_scaled, scaler = scale_data(X_train, X_validation)

        models = []
        models.append(('LR', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
        models.append(('CART', DecisionTreeClassifier(random_state=RANDOM_STATE)))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(kernel='linear', C=0.1, random_state=RANDOM_STATE)))

        results = []
        names = []
        print("----------------------------------------------------------------")
        print(f"Порівняння алгоритмів ({N_SPLITS}-кратна крос-валідація, метрика: Accuracy):")
        print("----------------------------------------------------------------")

        for name, model in models:
            if name in ['CART', 'NB', 'LDA']:
                X_data = X
                Y_data = y
            else:
                X_data = data_encoded.drop('income', axis=1)
                numeric_features = X_data.select_dtypes(include=np.number).columns.tolist()
                X_data[numeric_features] = scaler.transform(X_data[numeric_features])
                Y_data = y
            
            kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
            cv_results = cross_val_score(model, X_data, Y_data, cv=kfold, scoring='accuracy')
            
            results.append(cv_results)
            names.append(name)
            print('%s: %.4f (%.4f)' % (name, cv_results.mean(), cv_results.std()))

        print("----------------------------------------------------------------")

        plt.boxplot(results, labels=names)
        plt.title('Порівняння алгоритмів класифікації доходу')
        plt.ylabel('Accuracy Score (CV)')
        plt.show()

        best_index = np.argmax([r.mean() for r in results])
        best_name = names[best_index]
        best_score = results[best_index].mean()
        
        print(f"\n✅ Найкращий класифікатор (за середньою точністю CV): {best_name} з точністю {best_score:.4f}")

    except FileNotFoundError:
        print(f"Помилка: Файл '{INPUT_FILE}' не знайдено.")
    except Exception as e:
        print(f"Сталася помилка під час виконання: {e}")

if __name__ == "__main__":
    main()