# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SRIDHARAN J
RegisterNumber: 212222040158
```
```py
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Encoding:
![image](https://github.com/harini1006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497405/ed87456c-9dd8-418d-a960-1abad11477f2)


## Head():
![image](https://github.com/harini1006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497405/8e2c3fec-2fe3-40c3-923a-1a1c3719e734)


## Info():
![image](https://github.com/harini1006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497405/b48518c5-c983-44d3-9cc2-14924033aa91)


## isnull().sum():
![image](https://github.com/harini1006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497405/50754f89-e886-48c3-a285-44b76317b605)


## Prediction of y:
![image](https://github.com/user-attachments/assets/afb94e75-77da-4a4f-9eb6-009ed73fc17c)


## Accuracy:
![image](https://github.com/user-attachments/assets/676400a7-8dfb-4bf5-8c7a-6be670f6eee9)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
