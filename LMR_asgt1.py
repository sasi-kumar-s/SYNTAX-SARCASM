import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(9)
n_train = 50

x_train = np.sort(np.random.uniform(-3, 3, size=n_train)).reshape(-1, 1)
y_true_train = 0.5 * x_train.squeeze() ** 3 - 2 * x_train.squeeze()
y_train = y_true_train + np.random.normal(0, 3, size=n_train)

x_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_true_test = 0.5 * x_test.squeeze() ** 3 - 2 * x_test.squeeze()
y_test = y_true_test

degrees = [1, 2, 4, 8, 15]

train_errors = []
test_errors = []

plt.figure(figsize=(15,8))
for i,degree in enumerate(degrees):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    train_errors.append(mse_train)
    test_errors.append(mse_test)
    
    plt.subplot(2,3,i+1)
    plt.scatter(x_train, y_train, label="noisy data",alpha=0.7)
    plt.plot(x_test, y_true_test, label="True function")
    plt.plot(x_test,y_pred_test,label=f"degree:{degree}, test mse={mse_test:.1f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.suptitle("polynomial regression")
    plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(degrees, train_errors, marker='*', label="training error")
plt.plot(degrees, test_errors, marker='*', label="testing error")
plt.xlabel("degree")
plt.ylabel("mse")
plt.title("training vs testing error")
plt.legend()
plt.show()


for i, degree in enumerate(degrees):
    print(f"degree: {degree}, train mse: {train_errors[i]:.3f}, test mse: {test_errors[i]:.3f}")


# 1. Why does training error always decrease with higher polynomial degree?

# it will decrease because higher degree gives more flexibility to the model. if there is more flexibility then the model may gets overfitted.



# 2. Why does test error behave differently?

# it is because test data is unseen, when the degree starts increasing model starts fitting noise instead of real pattern.



# 3. At what point does the model start overfitting, and how can you tell?

# model starts overfitting when test error starts increasing while training error keeps decreasing. from the graph,this happens around higher polynomial degree.


# 4. If you had to choose one polynomial degree, which would it be and why?

# i choose 4th degree as it has low test error