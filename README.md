# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload the csv file and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeRegressor.
5. Import metrics and calculate the Mean squared error.
6. Apply metrics to the dataset, and predict the output.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Andra likitha
RegisterNumber:  21222122006
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print("data.head() for salary:")
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print("MSE value:")
mse

r2=metrics.r2_score(y_test,y_pred)
print("r2 value:")
r2

print("data prediction:")
dt.predict([[5,6]])
```

## Output:

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/e8bcc2fd-c601-4285-a91f-fb30d0cb969e)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/46bb92a3-605d-494f-a1eb-14d334506da9)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/b1055944-7529-4c71-8226-462443df69b0)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/94ab58fd-2682-49f1-8e5c-4626df806d06)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/7bdf803e-ae51-4bd2-bfae-0d0bb1e25c1c)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/594938b8-7611-4568-9721-2d5fa267726b)

![image](https://github.com/andralikitha/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/131592130/12756ca3-ab9d-4d68-970f-5655da07a112)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
