# Multiple Linear Regression With ATP Men's Data

The Purpose: Using the datas from Association of Tennis Professionals (ATP) Men's League from 2009 to 2017 to create exploratory data analysis and build a linear regression model to predict tennis match outcomes based on various statistics.

## Test Environment

Python 3.9.6

## Usage
Run the script using:
```python
python3 main.py
```
### Installation
To run this project, you need to have Python installed along with the following libraries:
numpy, pandas, seaborn, matplotlib, scikit-learn

You can import these libraries through Pip.

### Load Dataframe


The dataframe used in this project is contributed by @Sebastian-QR. The dataset is loaded from a CSV file and is slightly modified to increase the fluency of the project. 
The dataset contains various tennis statistics for different players. Key columns include:

Wins: Number of wins.
Losses: Number of losses.
Win%: Win percentage.
Other columns representing various player statistics (e.g., First serve, Aces, Double faults, etc.).

'Win%' is an added dependent variable and is used as the evaluator in this project. The dataset is cleaned by removing rows with missing 'Win%' values and duplicate entries based on the 'Player' column.
```Python
tennis_data = tennis_data.dropna(subset=['Win%'])
tennis_data = tennis_data.drop_duplicates(subset='Player')
```
### Exploratory Data Analysis (EDA)

Scatter plots and histograms are created for independent variables against 'Wins', 'Losses', 'Winnings', and 'Win%'. It visualize relationships each independent and outcome variables. 

### Prepare Data for Modeling

The lists of data from the Panda Dataframe was converted into numpy array. The dataset was split into training and test sets. 

```Python
tennis_data1_np = tennis_data.to_numpy()
X = tennis_data1_np[:, 2:-5] #disregard the first two colons of the dataset (Dates, Names)
y = tennis_data1_np[:, -1] (Only Win%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
```
### Evaluation of Model Performance 

The linear regression model is evaluated using the following metrics:
Mean Absolute Error (MAE): Measures the average magnitude of errors in a set of predictions.
Mean Squared Error (MSE): Measures the average of the squares of the errors.
R-squared score (R2): Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

```Python
Tennis_mse = mean_squared_error(y_test, Tennis_y_prediction)
Tennis_r2 = r2_score(y_test, Tennis_y_prediction)
Tennis_mae = mean_absolute_error(y_test, Tennis_y_prediction)
```

### Make a prediction

Create an example array and predict the outcome by:
```Python
Tennis_model_test = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
Tennis_pred = Tennis_model.predict(Tennis_model_test)
```

### Coefficient Analysis

The coefficients of the linear regression model are printed along with their corresponding variable names. A scatter plot comparing the actual and predicted values of 'Win%' is created to visualize the model's performance
```Python
coef = Tennis_model.coef_
Variable_names = ['First serve', 'First serve points won', 'First serve return points won', 'Second serve points won', 'Second serve return points won', 'Aces', 'Break points converted', 'Break points faced', 'Break points opportunities', 'Break points saved', 'Double faults', 'Return games played', 'Return games won', 'Return points won', 'Service games won', 'Total points won', 'Service games played','Total service points won']
for var_names, var_coef in zip(Variable_names, coef):
    print(f"{var_names}:{var_coef}")
plt.scatter(y_test, Tennis_y_prediction, color="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linewidth=3)
plt.xlabel("Tennis_Win%_Actual")
plt.ylabel("Tennis_Win%_Predicted")
plt.title("Tennis_Win%: Actual vs Predicted")
plt.show()
```
