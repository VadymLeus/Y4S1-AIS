import numpy as np
import pandas as pd
import re
import warnings
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler 

warnings.filterwarnings('ignore')

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
INPUT_FILE = 'income_data.txt'

def clean_data(data):
    data = data.replace(r'^\s*\?+\s*$', np.nan, regex=True).dropna()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str).str.strip().str.lower()
        data[col] = data[col].apply(lambda x: re.sub(r'[^a-z0-9<=->]', '', x))
    return data

def encode_categorical_data(data):
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

def evaluate_model(classifier, X_test, y_test, kernel_name):
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    print("\n" + "=" * 55)
    print(f"🔹 Оцінка Якості Моделі: {kernel_name} Ядро 🔹")
    print(f"Акуратність: {accuracy:.2f}%")
    print(f"Точність: {precision:.2f}%")
    print(f"Повнота: {recall:.2f}%")
    print(f"F1-міра: {f1:.2f}%")
    print("=" * 55)

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

def prepare_input_data(input_data, label_encoders):
    input_df = pd.DataFrame([input_data], columns=COLUMNS[:-1])
    input_data_encoded = []
    
    for col in COLUMNS[:-1]:
        item = str(input_df[col].iloc[0]).strip().lower()
        item = re.sub(r'[^a-z0-9<=->]', '', item)
        
        try:
            if col in label_encoders:
                encoded_val = label_encoders[col].transform([item])[0]
            else:
                encoded_val = int(item)
            input_data_encoded.append(encoded_val)
        except ValueError:
            print(f"Помилка кодування: '{item}' для '{col}'.")
            return None
    
    return pd.DataFrame([input_data_encoded], columns=COLUMNS[:-1])

def predict_income_scaled(classifier, label_encoders, input_data, scaler, numeric_features):
    X_input_df = prepare_input_data(input_data, label_encoders)
    if X_input_df is None: return None
    
    X_input_df[numeric_features] = scaler.transform(X_input_df[numeric_features])
    prediction = classifier.predict(X_input_df)
    return label_encoders['income'].inverse_transform(prediction)[0]

def main():
    KERNEL_NAME = "Поліноміальне"
    
    try:
        data = pd.read_csv(
            INPUT_FILE, 
            header=None, 
            names=COLUMNS, 
            sep=r'\s*,\s*', 
            engine='python', 
            na_values=['?']
        )
        
        data = clean_data(data)
        data_encoded, label_encoders = encode_categorical_data(data)

        X = data_encoded.drop('income', axis=1)
        y = data_encoded['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        
        classifier = SVC(kernel='poly', degree=3, random_state=0)
        print(f"Навчання SVM з {KERNEL_NAME} ядром...")
        classifier.fit(X_train, y_train) 
        print("Навчання завершено.")

        metrics = evaluate_model(classifier, X_test, y_test, KERNEL_NAME)
        
        X_scaled_full = X.copy()
        X_scaled_full[numeric_features] = scaler.transform(X_scaled_full[numeric_features])
        
        f1_cv_scores = cross_val_score(classifier, X_scaled_full, y, scoring='f1_weighted', cv=3)
        print(f"F1 score (Cross-Validation): {f1_cv_scores.mean() * 100:.2f}%")
        print("\n" + "=" * 55)

        test_input_data = [
            '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
            'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
            '0', '0', '40', 'United-States'
        ]
        
        print(f"\n🔹 Прогноз ({KERNEL_NAME} ядро) 🔹")
        
        predicted_income = predict_income_scaled(classifier, label_encoders, test_input_data, scaler, numeric_features)
        
        if predicted_income:
            print(f"Спрогнозований дохід: {predicted_income.upper()}")
            print("=" * 55)

    except FileNotFoundError:
        print(f"Файл '{INPUT_FILE}' не знайдено.")
    except Exception as e:
        print(f"Помилка: {e}")

if __name__ == "__main__":
    main()