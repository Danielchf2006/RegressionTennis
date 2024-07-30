import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from numpy.linalg import inv

tennis_data = pd.read_csv('tennis_stats_99.csv')
print(tennis_data.head())
tennis_data.info()
print(tennis_data.describe())
tennis_data = tennis_data.dropna(subset=['Win%'])
tennis_data = tennis_data.drop_duplicates(subset='Player')
#flts = [7,9,10,12,13,16,20,21,22,23]
#for flt in flts:
    #vu=tennis_data.columns[[flt]]
    #vu.iloc[1:] = vu.iloc[1:].astype(float)

vars = tennis_data.columns[[7,6,10,8,13,14,15,18,12,2,3,5,9,11,16,17,19]]
figure, axes = plt.subplots(len(vars), 5, figsize=(12.5, 30))
sns.set(font_scale=0.7)
figure.subplots_adjust(hspace=0.5, wspace=0.4)
for var in vars:
    sns.scatterplot(x=var, y='Wins', data=tennis_data, ax=axes[vars.tolist().index(var),0], alpha=0.4).set_title(f'{var} vs Wins', fontsize=7, weight='bold')
    sns.scatterplot(x=var, y='Losses', data=tennis_data, ax=axes[vars.tolist().index(var),1], alpha=0.4).set_title(f'{var} vs Losses', fontsize=7, weight='bold')
    sns.scatterplot(x=var, y='Winnings', data=tennis_data, ax=axes[vars.tolist().index(var), 2], alpha=0.4).set_title(f'{var} vs Winnings', fontsize=7, weight='bold')
    sns.scatterplot(x=var, y='Win%', data=tennis_data, ax=axes[vars.tolist().index(var), 3], alpha=0.4).set_title(f'{var} vs Winnings', fontsize=7, weight='bold')
    sns.histplot(x=var, data=tennis_data, ax=axes[vars.tolist().index(var),4]).set_title('Distribution', fontsize=7, weight='bold')
plt.show()
zvs = tennis_data.select_dtypes(include=[np.number])
print(zvs.corr()['Wins'])
print(zvs.corr()['Losses'])
print(zvs.corr()['Winnings'])

#tennis_data1 = tennis_data.drop(columns=['BreakPointsFaced', 'BreakPointsOpportunities', 'ReturnGamesPlayed', 'ServiceGamesPlayed'])
tennis_data1_np = tennis_data.to_numpy()
print(tennis_data1_np.shape)

X = tennis_data1_np[:, 2:-5]
y = tennis_data1_np[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
Tennis_model = LinearRegression().fit(x_train, y_train)
Tennis_y_prediction = Tennis_model.predict(x_test)
Tennis_mse = mean_squared_error(y_test, Tennis_y_prediction)
Tennis_r2 = r2_score(y_test, Tennis_y_prediction)
Tennis_mae = mean_absolute_error(y_test, Tennis_y_prediction)
print(f"Mean Absolute Error: {Tennis_mae:f}")
print(f"Mean squared error: {Tennis_mse:f}")
print(f"R-squared score: {Tennis_r2:f}")
#x_train, y_train = tennis_data1_np[:1292, 2:], tennis_data1_np[:1292, -1]
#x_test, y_test = tennis_data1_np[1292:1722, 2:], tennis_data1_np[1292:1722, -1]
#print(y_train.shape)
#print(x_train)



#M2nerror =("Mean squared error: %.2f" % mean_squared_error(x_test, Tennis_y_prediction))
#print(M2nerror)
#def get_predictions(model, X):
    #(n, p_neg1) = X.shape
    #p = p_neg1 + 1
    #nX = np.ones(shape=(n, p))
    #nX[:, 1:] = X

    #return np.dot(nX, Tennis_model)
Tennis_model_test = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
Tennis_pred = Tennis_model.predict(Tennis_model_test)
print(f"Prediction for Tennis model: {Tennis_pred[0]}")

coef = Tennis_model.coef_
Variable_names = ['First serve', 'First serve points won', 'First serve return points won', 'Second serve points won', 'Second serve return points won', 'Aces', 'Break points converted', 'Break points faced', 'Break points opportunities', 'Break points saved', 'Double faults', 'Return games played', 'Return games won', 'Return points won', 'Service games won', 'Total points won', 'Service games played','Total service points won']
for var_names, var_coef in zip(Variable_names, coef):
    print(f"{var_names}:{var_coef}")
plt.scatter(y_test, Tennis_y_prediction, color="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linewidth=3)
plt.xlabel("Tennis_Win%_Actual")
plt.ylabel("Tennis_Win%_Predicted")
plt.title("Tennis_WIn%:Actual vs Predicted")
plt.show()

