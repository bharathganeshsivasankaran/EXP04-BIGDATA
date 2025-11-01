# EXP04-BIGDATA

### 1) To implement multiple linear regression in python
### 2) To plot a normal distribution using matplotlib, numpy, histogram in python

```
Name : Bharathganesh S
Reg No : 212222230022
```

## AIM:

To implement Multiple Linear Regression in Python to predict a target variable using multiple independent variables and to plot a Normal Distribution using Matplotlib and NumPy by generating random data and visualizing it with a histogram and probability density curve.

## PROCEDURE:

Step : 1 Import Required Libraries – Import numpy, pandas, matplotlib, and necessary modules from sklearn for regression implementation and data visualization.

Step : 2 Prepare the Dataset – Create or load a dataset containing multiple independent variables (features) and one dependent variable (target).

Step : 3 Define Features and Target – Separate the dataset into input variables (X) and output variable (y).

Step : 4 Split the Data – Divide the dataset into training and testing sets using train_test_split() to evaluate model performance.

Step : 5 Train the Model – Use the LinearRegression() class from sklearn to train the model on the training data.

Step : 6 Predict the Output – Use the trained model to predict target values for the test data.

Step : 7 Evaluate Model Performance – Calculate metrics such as Mean Squared Error (MSE) and R² score to measure accuracy.

Step : 8 Generate Random Data for Normal Distribution – Use numpy.random.normal() to create random data points with a defined mean and standard deviation.

Step : 9 Plot Histogram and Curve – Use matplotlib.pyplot.hist() to plot the histogram and overlay the normal distribution curve using the probability density function formula.

Step : 10 Display Results – Show the regression output (coefficients, predictions, and accuracy) and visualize the normal distribution plot with proper titles, labels, and legends.

## PROGRAM:
### 1 - Multiple Linear Regression Implementation in Python
```py
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a sample dataset
data = {
    'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'x2': [2, 1, 4, 3, 5, 7, 6, 8, 9, 10],
    'y':  [5, 7, 9, 10, 13, 15, 17, 18, 21, 24]
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['x1', 'x2']]
y = df['y']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Compare predictions vs actual values
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPrediction Results:")
print(result)

```
### 2 - Plotting a Normal Distribution with Matplotlib, NumPy, and Histogram
```py
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate random data from a normal distribution
mu, sigma = 0, 1  # mean and standard deviation
data = np.random.normal(mu, sigma, 1000)  # 1000 random points

# Step 2: Plot histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')

# Step 3: Plot the theoretical normal distribution curve
xmin, xmax = plt.xlim()  # Get range for x-axis
x = np.linspace(xmin, xmax, 100)
p = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

plt.plot(x, p, 'r', linewidth=2, label='Normal Distribution Curve')

# Step 4: Add labels, title, legend
plt.title('Normal Distribution with Histogram')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

# Step 5: Show the plot
plt.show()

```
## OUTPUT:

### 1 - Multiple Linear Regression Implementation in Python

<img width="1075" height="217" alt="image" src="https://github.com/user-attachments/assets/ed6e9e39-1ee7-4ce0-bd0d-fa7e21a60557" />

### 2 - Plotting a Normal Distribution with Matplotlib, NumPy, and Histogram

<img width="1305" height="596" alt="image" src="https://github.com/user-attachments/assets/f166065b-fdf5-44e5-9c6b-596eeac92195" />

## RESULT:
 The Multiple Linear Regression model was successfully implemented to predict the target variable with good accuracy, and the Normal Distribution was effectively visualized using a histogram and probability density curve in Python.

