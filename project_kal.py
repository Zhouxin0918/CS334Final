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

warnings.filterwarnings("ignore")

quantity = pd.read_csv("Food_Supply_Quantity_kg_Data.csv")
quantity.head()

quantity.isnull().sum()
missing_col = [c for c in quantity.columns if quantity[c].isnull().sum() != 0]
print(missing_col)


# impute missing data
def impute_missing(df):
    # transform data from object variable to numerical variable
    df["Undernourished"] = (
        df["Undernourished"]
        .map(lambda x: "2.4" if x == "<2.5" else x)
        .astype("float64")
    )

    col = list(df.columns)
    col.remove("Unit (all except Population)")  # remove '%' unit
    col.remove("Country")  # object variable
    col.remove("Confirmed")
    col.remove("Deaths")
    col.remove("Recovered")
    col.remove("Active")
    # nan_col = [c for c in col if df[c].isnull().sum() > 0]
    # not_nan_col = [c for c in col if c not in nan_col]

    imputer = KNNImputer(n_neighbors=3)
    imp_data = imputer.fit_transform(df[col])
    new = pd.DataFrame(imp_data, columns=col)

    return pd.concat(
        [df[["Country"]], new, df[["Confirmed", "Deaths", "Recovered", "Active"]]],
        axis=1,
    )  # concat
    # 'Country' to data frame then return


quantity = impute_missing(quantity)


# quantity = quantity.dropna()
quantity = quantity.fillna(0)
print(missing_col)

fig, ax = plt.subplots(figsize=(20, 15))
mask = np.triu(np.ones_like(quantity.corr(), dtype=np.bool))
mask = mask[1:, :-1]
corr = quantity.corr().iloc[1:, :-1]

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8},
)
# plt.show()


# remove 'country', 'active' and 'recovered'
quantity = quantity.drop(columns=["Active", "Country", "Recovered"])

# top 12 variables related to confirmed from each dataset
quantity = quantity[
    abs(quantity.corr()["Confirmed"]).sort_values(ascending=False)[:12].index
]
print("aaaaaa")
print("quantity")
variables_quantity = [c for c in quantity.columns]
print(variables_quantity)

# top 12 variables related to deaths from each dataset

quantityD = quantity[
    abs(quantity.corr()["Deaths"]).sort_values(ascending=False)[:12].index
]
variables_quantityD = [c for c in quantity.columns]
print(variables_quantityD)

# remove variables being predicted
quantity_d = quantity.drop(columns=["Confirmed", "Deaths"])

# concatenate top variables related to from each dataset
x = np.asarray(quantity_d)
print("selected variables")
print(x.shape[1])

y = quantity[["Confirmed", "Deaths"]]

# splitting
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# lienar regression
print("-------Predicting confirmed cases using LinearRegression-------")
line_model = LinearRegression()
line_model = line_model.fit(X_train, y_train["Confirmed"])
pred_train_line = line_model.predict(X_train)
pred_test_line = line_model.predict(X_test)
coefficient = pd.DataFrame(line_model.coef_)
print(coefficient.columns)
print("\nCoefficient of model :", coefficient)
print("-------evaluation on training data-------")
print(
    "the MAE of line model in Confirmed testing data:",
    mean_absolute_error(y_train["Confirmed"], pred_train_line),
)
print(
    "the MSE of line model in Confirmed testing data:",
    mean_squared_error(y_train["Confirmed"], pred_train_line),
)
print(
    "the RMSE of line model in confirmed testing data:",
    sqrt(mean_squared_error(y_train["Confirmed"], pred_train_line)),
)
print("Variance score: %.2f" % r2_score(y_train["Confirmed"], pred_train_line))
print("-------evaluation on testing data-------")
print(
    "the MAE of line model in Confirmed testing data:",
    mean_absolute_error(y_test["Confirmed"], pred_test_line),
)
print(
    "the MSE of line model in Confirmed testing data:",
    mean_squared_error(y_test["Confirmed"], pred_test_line),
)
print(
    "the RMSE of line model in confirmed testing data:",
    sqrt(mean_squared_error(y_test["Confirmed"], pred_test_line)),
)
print("Variance score: %.2f" % r2_score(y_test["Confirmed"], pred_test_line))

print("-------Predicting deaths cases using LinearRegression-------")
line_model = line_model.fit(X_train, y_train["Deaths"])
pred_test_line = line_model.predict(X_test)
pred_train_line = line_model.predict(X_train)
coefficient = pd.DataFrame(line_model.coef_)
# print(coefficient.columns)
print("\nCoefficient of model :", coefficient)
print("-------evaluation on training data-------")
print(
    "the MSE of line model in deaths testing data:",
    mean_squared_error(y_train["Deaths"], pred_train_line),
)
print(
    "the MAE of line model in deaths testing data:",
    mean_absolute_error(y_train["Deaths"], pred_train_line),
)
print(
    "the RMSE of line model in deaths tetsing data:",
    sqrt(mean_squared_error(y_train["Deaths"], pred_train_line)),
)
print("Variance score: %.2f" % r2_score(y_train["Deaths"], pred_train_line))


print("-------evaluation on testing data-------")
print(
    "the MSE of line model in deaths testing data:",
    mean_squared_error(y_test["Deaths"], pred_test_line),
)
print(
    "the MAE of line model in deaths testing data:",
    mean_absolute_error(y_test["Deaths"], pred_test_line),
)
print(
    "the RMSE of line model in deaths tetsing data:",
    sqrt(mean_squared_error(y_test["Deaths"], pred_test_line)),
)

print("Variance score: %.2f" % r2_score(y_test["Deaths"], pred_test_line))

print("-------LASSO-------")
print("-------Confirmed cases-------")
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train["Confirmed"])
pred_train_lasso = model_lasso.predict(X_train)
coefficient = pd.DataFrame(model_lasso.coef_)
# print(coefficient.columns)
print("\nCoefficient of model :", coefficient)

print(
    "RMSE of LASSO model in confirmed trained data: ",
    np.sqrt(mean_squared_error(y_train["Confirmed"], pred_train_lasso)),
)
print(
    "variance score in trained data: ", r2_score(y_train["Confirmed"], pred_train_lasso)
)

pred_test_lasso = model_lasso.predict(X_test)
print(
    "RMSE of LASSO model in confirmed testing data: ",
    np.sqrt(mean_squared_error(y_test["Confirmed"], pred_test_lasso)),
)
print(
    "variance score in testing data: ", r2_score(y_test["Confirmed"], pred_test_lasso)
)

print("-------Death cases-------")
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train["Deaths"])
pred_train_lasso = model_lasso.predict(X_train)
coefficient = pd.DataFrame(model_lasso.coef_)
# print(coefficient.columns)
print("\nCoefficient of model :", coefficient)


print(
    "RMSE of LASSO model in death trained data: ",
    np.sqrt(mean_squared_error(y_train["Deaths"], pred_train_lasso)),
)
print("variance score in trained data: ", r2_score(y_train["Deaths"], pred_train_lasso))

pred_test_lasso = model_lasso.predict(X_test)
print(
    "RMSE of LASSO model in death testing data: ",
    np.sqrt(mean_squared_error(y_test["Deaths"], pred_test_lasso)),
)
print("variance score in testing data: ", r2_score(y_test["Deaths"], pred_test_lasso))

# KNN
# KNN
# parameters_KNN = {"n_neighbors": list(range(1, X_train.shape[0] - 1, 2))}
# knn = KNeighborsRegressor()
# best_knn = GridSearchCV(knn, parameters_KNN)
print("-------Predicting confirmed cases using KNN-------")
rmse_val = []  # to store rmse values for different k
for K in range(100):
    K = K + 1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    model.fit(X_train, y_train["Confirmed"])  # fit the model
    pred = model.predict(X_test)  # make prediction on test set
    error = sqrt(mean_squared_error(y_test["Confirmed"], pred))  # calculate rmse
    rmse_val.append(error)  # store rmse values
    print("RMSE value for k= ", K, "is:", error)
# plotting the rmse values against k values
curve = pd.DataFrame(rmse_val)  # elbow curve
# print("plotting")
curve.plot(title="finding best K")
plt.xlabel("K neighbors")
plt.ylabel("RMSE")
plt.savefig("elbow curve finding best K for predicting Death.png")
# print("plotting done!")

best_knn = neighbors.KNeighborsRegressor(n_neighbors=25)
best_knn.fit(X_train, y_train["Confirmed"])  # fit the model
pred_test_knn = best_knn.predict(X_test)  # make prediction on test set
error = sqrt(mean_squared_error(y_test["Confirmed"], pred_test_knn))  # calculate rmse
print("RMSE value for k= ", 25, "is:", error)
print(
    "the MSE of KNN model in Confirmed:",
    mean_squared_error(y_test["Confirmed"], pred_test_knn),
)
print(
    "the MAE of KNN model in Confirmed:",
    mean_absolute_error(y_test["Confirmed"], pred_test_knn),
)

print("-------Predicting Deaths using KNN-------")
rmse_val = []  # to store rmse values for different k
for K in range(100):
    K = K + 1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    model.fit(X_train, y_train["Deaths"])  # fit the model
    pred = model.predict(X_test)  # make prediction on test set
    error = sqrt(mean_squared_error(y_test["Deaths"], pred))  # calculate rmse
    rmse_val.append(error)  # store rmse values
    print("RMSE value for k= ", K, "is:", error)
# plotting the rmse values against k values
curve = pd.DataFrame(rmse_val)  # elbow curve
# print("plotting")
curve.plot(title="finding best K")
plt.xlabel("K neighbors")
plt.ylabel("RMSE")
plt.savefig("elbow curve finding best K for predicting Death.png")
# print("plotting done!")

best_knn = neighbors.KNeighborsRegressor(n_neighbors=23)
best_knn.fit(X_train, y_train["Deaths"])  # fit the model
pred_test_knn = best_knn.predict(X_test)  # make prediction on test set
error = sqrt(mean_squared_error(y_test["Deaths"], pred_test_knn))  # calculate rmse
print("RMSE value for k= ", 23, "is:", error)
print(
    "the MSE of KNN model in deaths:",
    mean_squared_error(y_test["Deaths"], pred_test_knn),
)
print(
    "the MAE of KNN model in deaths:",
    mean_absolute_error(y_test["Deaths"], pred_test_knn),
)

# Random Forest
print("-------Predicting confirmed cases using RandomForest-------")
parameters_RF = {
    "n_estimators": list(range(5, 100, 5)),
    "criterion": ["squared_error", "absolute_error", "poisson"],
    "max_depth": list(range(1, 25)),
    "min_samples_split": list(range(1, X_train.shape[0], 2)),
    "min_samples_leaf": list(range(1, X_train.shape[0], 2)),
    "max_features": list(range(1, 25)),
}
rf = RandomForestRegressor()
best_rf = RandomizedSearchCV(
    rf, parameters_RF, cv=5, return_train_score=False, n_iter=10
)
best_rf.fit(X_train, y_train["Confirmed"])
print(best_rf.cv_results_)
pred_test_rf = best_rf.predict(X_test)
print(
    "the MSE of RF model in Confirmed:",
    mean_squared_error(y_test["Confirmed"], pred_test_rf),
)
print(
    "the MAE of RF model in Confirmed:",
    mean_absolute_error(y_test["Confirmed"], pred_test_rf),
)
print(
    "the RMSE of RF model in Confirmed:",
    sqrt(mean_squared_error(y_test["Confirmed"], pred_test_rf)),
)
print("-------Predicting death cases using RandomForest-------")
best_rf.fit(X_train, y_train["Deaths"])
pred_test_rf = best_rf.predict(X_test)
print(
    "the MSE of RF model in death:", mean_squared_error(y_test["Deaths"], pred_test_rf)
)
print(
    "the MAE of RF model in death:", mean_absolute_error(y_test["Deaths"], pred_test_rf)
)
print(
    "the MSE of RF model in death:",
    sqrt(mean_squared_error(y_test["Deaths"], pred_test_rf)),
)
