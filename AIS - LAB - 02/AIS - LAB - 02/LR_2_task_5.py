import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from io import BytesIO 
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef

iris = load_iris()
X, y = iris.data, iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)

clf = RidgeClassifier(
    tol = 1e-2,
    solver = "sag",
    random_state = 0
)

clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)
print("="*40)
print("Результати класифікатора Ridge")
print("="*40)

print('Accuracy:', np.round(accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(precision_score(ytest, ypred, average = 'weighted'), 4))
print('Recall:', np.round(recall_score(ytest, ypred, average = 'weighted'), 4))
print('F1 Score:', np.round(f1_score(ytest, ypred, average = 'weighted'), 4))

print('Cohen Kappa Score:', np.round(cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corrcoef:', np.round(matthews_corrcoef(ytest, ypred), 4))

print('\n\t\tClassification Report:\n', metrics.classification_report(ytest, ypred)) 

mat = confusion_matrix(ytest, ypred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    mat.T, 
    square = True, 
    annot = True, 
    fmt = 'd', 
    cbar = False,
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel('Істинні мітки (True Label)')
plt.ylabel('Прогнозовані мітки (Predicted Label)');
plt.title('Матриця плутанини (Ridge Classifier)')

plt.savefig("Confusion.jpg") 
print("\nМатриця плутанини збережена як Confusion.jpg")

f = BytesIO()
plt.savefig(f, format = "svg")