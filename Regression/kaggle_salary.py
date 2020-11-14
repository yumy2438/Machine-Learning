import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from regression_imp import *
# salary data at kaggle: https://www.kaggle.com/rsadiq/salary
salary_data = pd.read_csv("../datasets/kaggle-salary.csv")
salary_data = salary_data.values

years = salary_data[:, 0].reshape(-1, 1)  # input
salaries = salary_data[:, 1].reshape(-1, 1)  # output

# train validation split
x_train, x_valid, y_train, y_valid = train_test_split(years, salaries)

# my regression implementation
regression_model = Regression(x_train.T, y_train.T)
regression_model.train_linear_model("GD", 1e-2, 3000)
# checking gradient descent working correctly.
plt.plot(np.linspace(1, 3000, 3000), regression_model.cost_history)
plt.show()

# predictions on training data
my_pred_train = regression_model.predict_output().T
# predictions on validation data
my_pred_test = regression_model.predict_output_with_given_features(x_valid.T)

# linear regression model from sklearn library
model = LinearRegression()
model.fit(x_train, y_train)
sklearn_pred_train = model.predict(x_train)
sklearn_pred_test = model.predict(x_valid)

# mean absolute errors on test data
print(mean_absolute_error(y_valid, my_pred_test))
print(mean_absolute_error(y_valid, sklearn_pred_test))

# comparing my implementation with sklearn
plt.plot(x_valid, y_valid, '.')
plt.plot(x_valid, my_pred_test, 'b-')
plt.plot(x_valid, sklearn_pred_test, 'g-')
plt.plot(x_train, y_train, '.')
plt.plot()
plt.show()