try:
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import re
    import warnings
except ImportError as e:
    print(f"Помилка: Не вдалося імпортувати необхідні бібліотеки: {e}")
    print("Будь ласка, встановіть їх за допомогою: pip install numpy pandas scikit-learn")
    exit(1)

warnings.filterwarnings('ignore')

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
INPUT_FILE = 'income_data.txt'


def clean_data(data):
    """
    Очистка даних: видалення зайвих пробілів, переведення у нижній регістр, 
    видалення рядків з '?' (відсутні значення).
    """
    # Заміна '?' на NaN і видалення рядків з NaN
    data = data.replace(r'^\s*\?+\s*$', np.nan, regex=True).dropna()

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str).str.strip().str.lower()
        data[col] = data[col].apply(lambda x: re.sub(r'[^a-z0-9<=->]', '', x))

    return data


def encode_categorical_data(data):
    """Кодування категоріальних змінних за допомогою LabelEncoder"""
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


def evaluate_model(classifier, X_test, y_test):
    """
    Прогноз та оцінка якості моделі на тестовій вибірці.
    """
    y_pred = classifier.predict(X_test)
    
    # Обчислення метрик
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    print("\n" + "=" * 48)
    print("🔹 Оцінка Якості Моделі на Тестовій Вибірці 🔹")
    print(f"Акуратність (Accuracy): {accuracy:.2f}%")
    print(f"Точність (Precision): {precision:.2f}%")
    print(f"Повнота (Recall): {recall:.2f}%")
    print(f"F1-міра (F1 Score): {f1:.2f}")
    print("=" * 48)

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}


def prepare_input_data(input_data, label_encoders):
    """Підготовка вхідних даних для прогнозування"""

    input_df = pd.DataFrame([input_data], columns=COLUMNS[:-1])
    
    input_data_encoded = []
    
    for col in COLUMNS[:-1]:
        item = str(input_df[col].iloc[0]).strip().lower()
        item = re.sub(r'[^a-z0-9<=->]', '', item)
        
        try:
            if col in label_encoders:
                # Категоріальна ознака
                encoded_val = label_encoders[col].transform([item])[0]
            else:
                # Числова ознака
                encoded_val = int(item)
            input_data_encoded.append(encoded_val)
        except ValueError as e:
            print(f"Помилка кодування: Значення '{item}' не знайдено для ознаки '{col}'.")
            return None
    
    return pd.DataFrame([input_data_encoded], columns=COLUMNS[:-1])


def predict_income(classifier, label_encoders, input_data):
    """Прогнозування доходу для нових даних"""

    X_input = prepare_input_data(input_data, label_encoders)
    if X_input is None:
        return None

    # Прогноз
    prediction = classifier.predict(X_input)

    # Декодування рез
    predicted_income = label_encoders['income'].inverse_transform(prediction)[0]
    
    return predicted_income


def main():
    """Головна функція програми"""
    try:
        # 1. Завантаження та попередня обробка даних
        data = pd.read_csv(
            INPUT_FILE, 
            header=None, 
            names=COLUMNS, 
            sep=r'\s*,\s*', 
            engine='python', 
            na_values=['?']
        )
        
        data = clean_data(data)
        
        # 2. Кодування категоріальних ознак
        data_encoded, label_encoders = encode_categorical_data(data)

        # 3. Розділення на ознаки та цільову змінну
        X = data_encoded.drop('income', axis=1)
        y = data_encoded['income']

        # 4. Розділення на навчальну та тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=5
        )

        # 5. Створення та навчання класифікатора
        classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual="auto", max_iter=10000))
        print("Починається навчання класифікатора...")
        classifier.fit(X_train, y_train)
        print("Навчання завершено.")

        # 6. Оцінка моделі
        metrics = evaluate_model(classifier, X_test, y_test)
        
        # Крос-валідація
        f1_cv_scores = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
        print(f"F1 score (Cross-Validation, cv=3): {f1_cv_scores.mean() * 100:.2f}%")
        print("\n" + "=" * 40)

        # 7. Прогноз для тестової точки даних
        test_input_data = [
            '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
            'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
            '0', '0', '40', 'United-States'
        ]
        
        print("\n🔹 Прогноз для тестової точки даних 🔹")
        print(f"Вхідні дані: {test_input_data}")
        
        predicted_income = predict_income(classifier, label_encoders, test_input_data)
        
        if predicted_income:
            print(f"Спрогнозований клас доходу: {predicted_income.upper()}")
            print("=" * 40)

    except FileNotFoundError:
        print(f"Помилка: Файл '{INPUT_FILE}' не знайдено.")
    except Exception as e:
        print(f"Сталася помилка під час виконання: {e}")


if __name__ == "__main__":
    main()