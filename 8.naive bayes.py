import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
dataset = pd.read_csv("C:/Users/Acer/Desktop/DEEP LEARNING/IRIS.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
nb_y_pred = naive_bayes_classifier.predict(X_test)
nb_cm = confusion_matrix(y_test, nb_y_pred)
nb_accuracy = accuracy_score(y_test, nb_y_pred)

print("Naive Bayes - Confusion Matrix:")
print(nb_cm)
print("Naive Bayes - Accuracy:", nb_accuracy)

logistic_classifier = LogisticRegression(random_state=0)
logistic_classifier.fit(X_train, y_train)
logistic_y_pred = logistic_classifier.predict(X_test)
logistic_cm = confusion_matrix(y_test, logistic_y_pred)
logistic_accuracy = accuracy_score(y_test, logistic_y_pred)

print("Logistic Regression - Confusion Matrix:")
print(logistic_cm)
print("Logistic Regression - Accuracy:", logistic_accuracy)
