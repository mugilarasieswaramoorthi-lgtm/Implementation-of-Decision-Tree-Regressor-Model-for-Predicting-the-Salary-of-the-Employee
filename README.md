# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Mugilarasi E 
RegisterNumber:25017644

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Education_Level': [1, 2, 1, 3, 2, 2, 3, 1, 3, 2],  # 1=Bachelor, 2=Master, 3=PhD
    'Department': [1, 2, 3, 2, 1, 3, 1, 2, 3, 2],      # 1=HR, 2=IT, 3=Finance
    'Salary': [30000, 35000, 32000, 40000, 38000, 45000, 50000, 42000, 52000, 60000]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

X = df[['Experience', 'Education_Level', 'Department']].values
y = df['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

new_employee = np.array([[5, 2, 3]]) 
predicted_salary = regressor.predict(new_employee)
print(f"Predicted Salary for new employee: {predicted_salary[0]}")

new_employee = np.array([[7]]) 
predicted_salary = regressor.predict(new_employee)
print(f"Predicted Salary for {new_employee[0][0]} years of experience: {predicted_salary[0]}")


```

## Output:
<img width="758" height="362" alt="Screenshot 2025-10-06 210931" src="https://github.com/user-attachments/assets/aaf4b19c-9bf3-486b-9fa2-74b8ff1a235b" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
