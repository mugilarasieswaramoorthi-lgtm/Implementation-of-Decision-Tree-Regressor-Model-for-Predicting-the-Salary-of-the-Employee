# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start and Import the Required Libraries

2.Load and Prepare the Dataset

3.Split the Dataset into Training and Testing Sets

4.Train the Decision Tree Regressor and Make Predictions

5.Evaluate Model Performance and Predict New Employee Salaries 

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Mugilarasi E 
RegisterNumber:25017644

import pandas as pd 
data=pd.read_csv("Salary.csv") 
data.head() 
data.info() 
data.isnull().sum() 
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
data["Position"]=le.fit_transform(data["Position"]) 
data.head() 
x=data[["Position","Level"]] 
x.head() 
y=data["Salary"] 
y.head() 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2) 
from sklearn.tree import DecisionTreeRegressor 
dt=DecisionTreeRegressor() 
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test) 
print(y_pred )
 mse=metrics.mean_squared_error(y_test, y_pred)
 print(mse)
 r2= metrics.r2_score(y_test,y_pred)
 print(r2)
 dt.predict([[5,6]])


```

## Output:
![9th 1](https://github.com/user-attachments/assets/d87e5eb4-7c3d-4aad-8060-3e8b39860dda)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
