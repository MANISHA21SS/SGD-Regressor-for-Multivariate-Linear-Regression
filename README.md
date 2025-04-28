# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import necessary libraries
Step 2: Load and explore the dataset
Step 3: Preprocess the data
Step 4: Split the data
Use train_test_split() to divide the dataset into training and testing sets (e.g., 80% training, 20% testing).
Step 5: Create and train the SGD Regressor model
Step 6: Predict on the test data
Use the trained models to predict y_pred for test features X_test.
Step 7: Evaluate the model
Compare the predicted values (y_pred) with the actual values (y_test).
Step 8: Display the results
Print the actual vs predicted results.
Print evaluation metrics to understand model performance.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Manisha selvakumari.S.S.
RegisterNumber: 212223220055  
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target

print("Name: Manisha selvakumari.S.S.")
print("Reg No: 212223220055")

print(df.head())

X=df.drop(columns=['AveOccup','HousingPrice'])
Y=df[['AveOccup','HousingPrice']]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train = scaler_X.fit_transform(X_train) 
X_test = scaler_X.transform(X_test) 
Y_train = scaler_Y.fit_transform(Y_train) 
Y_test = scaler_Y.transform(Y_test)

sgd =  SGDRegressor(max_iter=1000, tol=1e-3)

multi_output_sgd= MultiOutputRegressor(sgd) 
multi_output_sgd.fit(X_train, Y_train)

Y_pred=multi_output_sgd.predict(X_test)

Y_test= scaler_Y.inverse_transform(Y_pred)

mse= mean_squared_error (Y_test, Y_pred) 
print("Mean Squared Error:", mse) 

print("\nPredictions: \n",Y_pred[:5])
```

## Output:
![Screenshot 2025-04-28 210309](https://github.com/user-attachments/assets/175d2261-b64b-4a85-b7f3-59763a7387cb)
![Screenshot 2025-04-28 210331](https://github.com/user-attachments/assets/24c7d699-e6d3-47e2-b816-aaefca8cfc81)
![Screenshot 2025-04-28 210348](https://github.com/user-attachments/assets/76b619e2-7462-4928-8a66-1a66f511117e)
![Screenshot 2025-04-28 210354](https://github.com/user-attachments/assets/1bc44790-dcf3-47d0-98b8-0e5f702cb1d0)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
