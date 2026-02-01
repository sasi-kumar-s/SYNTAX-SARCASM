import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error


# 0 -> avg income        = avg income in lakhs
# 1 -> house age         = avg age of housesin yrs
# 2 -> rooms             = avg number of rooms
# 3 -> population        = population in the locality
# 4 -> distance to city  = distance from city center in km

# house_price = house price

x = np.array([
    [6.5, 10, 5, 3000, 8],
    [7.2, 8, 6, 2500, 5],
    [5.8, 15, 4, 4000, 12],
    [8.0, 5, 7, 2000, 3],
    [6.0, 20, 4, 4500, 15],
    [7.5, 7, 6, 2200, 4],
    [5.5, 25, 3, 4800, 18],
    [8.2, 4, 7, 1800, 2],
    [6.8, 12, 5, 3200, 7],
    [7.9, 6, 6, 2100, 4],
    [5.9, 18, 4, 4100, 14],
    [8.5, 3, 8, 1700, 1],
])

y = np.array([
    65, 72, 55, 85, 50, 75,
    45, 90, 68, 80, 52, 95
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

linear_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("linear", LinearRegression())
])

ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1))
])

elastic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5))
])

linear_pipe.fit(x_train, y_train)
ridge_pipe.fit(x_train, y_train)
lasso_pipe.fit(x_train, y_train)
elastic_pipe.fit(x_train, y_train)

print("linear:", mean_squared_error(y_test, linear_pipe.predict(x_test)))

print("ridge:", mean_squared_error(y_test, ridge_pipe.predict(x_test)))

print("lasso:", mean_squared_error(y_test, lasso_pipe.predict(x_test)))

print("elastic net:", mean_squared_error(y_test, elastic_pipe.predict(x_test)))

plt.figure(figsize=(10,5))

plt.plot(linear_pipe.steps[-1][1].coef_, label="linear")
plt.plot(ridge_pipe.steps[-1][1].coef_, label="ridge")
plt.plot(lasso_pipe.steps[-1][1].coef_, label="lasso")
plt.plot(elastic_pipe.steps[-1][1].coef_, label="elastic net")

plt.xlabel("index")
plt.ylabel("coefficient")
plt.title("coefficient shrinkage")
plt.legend()
plt.show()

ridge_low = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=0.01))
])

ridge_high = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=100))
])

ridge_low.fit(x_train, y_train)
ridge_high.fit(x_train, y_train)

plt.figure(figsize=(8,4))
plt.plot(ridge_low.steps[-1][1].coef_, label="alpha = 0.01")
plt.plot(ridge_high.steps[-1][1].coef_, label="alpha = 100")

plt.xlabel("index")
plt.ylabel("coefficient")
plt.title("ridge regularization")
plt.legend()
plt.show()


#1. Why does regularization improve test performance?

# it improve test performance because it won't give the chance to learn too much from training data, without regularization the model will fit in every point and the model will be overfitted.



#2. Why does Ridge keep all features but shrink them?

# it keep all features because it cannot make the coefficient zero.it will only reduce the size of coefficient.



#3. Why does Lasso remove features entirely?

# Lasso remove features because it push some coefficients to exactly zero. it is because it thinks some of the features are useless.



#4. Why does Elastic Net behave differently from both?

# it is combination of ridge and lasso. it shrink coefficients like ridge and remove features like lasso.




#5. Which model performed best on your dataset and why?

# Elastic net performed best on this dataset because elastic net error is lower than linear and lasso and similar to ridge.