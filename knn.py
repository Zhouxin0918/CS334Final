import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score as acc
import statsmodels.api as sm
import sklearn.metrics as metrics
from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.linear_model import Lasso

Food_Supply_Quantity_kg_dataset = pd.read_csv("Food_Supply_Quantity_kg_Data.csv")
Food_Supply_Quantity_kg_data = Food_Supply_Quantity_kg_dataset.fillna(0)
Food_Supply_Quantity_kg = Food_Supply_Quantity_kg_data.replace("<2.5", 2.5)

X = Food_Supply_Quantity_kg[
    [
        "Alcoholic Beverages",
        "Animal fats",
        "Animal Products",
        "Aquatic Products, Other",
        "Cereals - Excluding Beer",
        "Eggs",
        "Fish, Seafood",
        "Fruits - Excluding Wine",
        "Meat",
        "Milk - Excluding Butter",
        "Miscellaneous",
        "Offals",
        "Oilcrops",
        "Pulses",
        "Spices",
        "Starchy Roots",
        "Stimulants",
        "Sugar & Sweeteners",
        "Sugar Crops",
        "Treenuts",
        "Vegetable Oils",
        "Vegetables",
        "Vegetal Products",
        "Obesity",
        "Population",
    ]
]

y = Food_Supply_Quantity_kg["Deaths"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN
# KNN
# parameters_KNN = {"n_neighbors": list(range(1, X_train.shape[0] - 1, 2))}
# knn = KNeighborsRegressor()
# best_knn = GridSearchCV(knn, parameters_KNN)


# print("-------Predicting Deaths using KNN-------")
# rmse_val = []  # to store rmse values for different k
# for K in range(100):
#     K = K + 1
#     model = neighbors.KNeighborsRegressor(n_neighbors=K)

#     model.fit(X_train, y_train)  # fit the model
#     pred = model.predict(X_test)  # make prediction on test set
#     error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
#     rmse_val.append(error)  # store rmse values
#     print("RMSE value for k= ", K, "is:", error)
# # plotting the rmse values against k values
# curve = pd.DataFrame(rmse_val)  # elbow curve
# # print("plotting")
# curve.plot(title="finding best K")
# plt.xlabel("K neighbors")
# plt.ylabel("RMSE")
# plt.savefig("elbow curve finding best K for predicting Death.png")
# # print("plotting done!")

# best_knn = neighbors.KNeighborsRegressor(n_neighbors=23)
# best_knn.fit(X_train, y_train)  # fit the model
# pred_test_knn = best_knn.predict(X_test)  # make prediction on test set
# error = sqrt(mean_squared_error(y_test, pred_test_knn))  # calculate rmse
# print("RMSE value for k= ", 23, "is:", error)
# print(
#     "the MSE of KNN model in deaths:",
#     mean_squared_error(y_test, pred_test_knn),
# )
# print(
#     "the MAE of KNN model in deaths:",
#     mean_absolute_error(y_test, pred_test_knn),
# )


rmse_val = []  # to store rmse values for different k
for K in range(100):
    K = K + 1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    model.fit(X_train, y_train)  # fit the model
    pred = model.predict(X_test)  # make prediction on test set
    error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
    rmse_val.append(error)  # store rmse values
    print("RMSE value for k= ", K, "is:", error)
