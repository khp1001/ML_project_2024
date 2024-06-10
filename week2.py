import numpy as np
import pandas as pd

cancer_dataset = pd.read_csv('D:/YEAR 2/ML camp/breast+cancer+coimbra/dataR2.csv')

cancer_dataset.head()

cancer_dataset.tail()

"""Linear Regression"""

#Write Code for Logistic Regression here

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = cancer_dataset.drop(columns=['Classification'])
Y = cancer_dataset['Classification']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)
print("Logistic Regression Accuracy:", clf.score(X_test, Y_test))


"""SVM(Support Vector Machines)"""


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X = cancer_dataset.drop(columns=['Classification'])
Y = cancer_dataset['Classification']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = SVC(kernel = 'linear', C=3)
model.fit(X_train, Y_train)
print("SVM Accuracy:", model.score(X_test, Y_test))


"""**LINEAR REGRESSION**"""

#For more information for the dataser refer: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
real_estate_dataset = pd.read_excel('D:/YEAR 2/ML camp/real+estate+valuation+data+set/Real estate valuation data set.xlsx')

real_estate_dataset.head()

real_estate_dataset.tail()

#Code for Linear Regression here
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = real_estate_dataset.drop(columns=['Y house price of unit area'])
Y = real_estate_dataset['Y house price of unit area']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X_train, Y_train)
print("Linear Regression Accuracy:", model2.score(X_test, Y_test))