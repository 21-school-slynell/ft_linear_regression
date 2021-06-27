#!/usr/bin/env python3
from libs import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

SPEED = 0.1
EPS = 0.000000001

df = pd.DataFrame()
try:
    df = pd.read_csv("./data/data.csv")
except OSError:
    snackbar("Error: can\'t open data file (data/data.csv)", 'error')
except (ValueError, IndexError):
    snackbar("Error: not valid data file (data/data.csv)", 'error')
except Exception as err:
    snackbar("Unknown error", 'error')

if (df.shape[1] != 2):
    snackbar("There are two features missing from your dataset", 'error')

dt = pd.DataFrame()
try:
    dt = pd.read_csv("./model.csv")
except OSError:
    snackbar("Error: can\'t open model file (model.csv)", 'info')
except (ValueError, IndexError):
    snackbar("Error: not valid model file (model.csv)", 'info')
except Exception as err:
    snackbar("Unknown error", 'info')

if (list(dt.shape) != [1, 2]):
    dt['k'] = [0]
    dt['b'] = [0]

b = dt['b'][0]
k = dt['k'][0]

X, Y = [df.km, df.price]

def predict(b, k, x):
    return k * x + b


Y_predic_my = predict(b, k, X)
df['my_price'] = Y_predic_my


def MAE(Y_predic, Y):
    return (abs(Y_predic - Y)).mean()


def MSE(Y_predic, Y):
    return ((Y_predic - Y) ** 2).mean()


def R2(Y_predic, Y):
    y_mean = Y.mean()
    return 1 - ((Y_predic - Y) ** 2).sum() / ((Y - y_mean) ** 2).sum()

def unzip (arr):
    return list(map(lambda q: q[0], arr))

linear_regressor = LinearRegression()
X_normal, Y_normal = [
    df.iloc[:, 0].values.reshape(-1, 1), df.iloc[:, 1].values.reshape(-1, 1)]
linear_regressor.fit(X_normal, Y_normal)
Y_pred_sc = linear_regressor.predict(X_normal)
df['sc_price'] = Y_pred_sc
b_sc = unzip(linear_regressor.predict([[0]]))[0]
k_sc = linear_regressor.coef_[0][0]

snackbar('My model:', 'info')
print('\t MSE = {metric}'.format(metric=MSE(df['my_price'], df['price'])))
print('\t MAE = {metric}'.format(metric=MAE(df['my_price'], df['price'])))
print('\t R2(metric) = {metric}'.format(metric=R2(df['my_price'], df['price'])))
print()
snackbar('Sclearn model:', 'info')
print('\t MSE = {metric}'.format(metric=MSE(df['sc_price'], df['price'])))
print('\t MAE = {metric}'.format(metric=MAE(df['sc_price'], df['price'])))
print('\t R2(metric) = {metric}'.format(metric=R2(df['sc_price'], df['price'])))


x, y = np.array(df.km), np.array(df.price)
score = []
n_epoche = list(range(500,4000, 50))
for n in n_epoche:
	model = SlyLinearRegression(0, 0)
	model.fit(x, y, n)
	score.append(model.score(x, y))


fig, axs = plt.subplots(2, 1)

axs[0].plot(df.km, df.price, '*')
axs[0].plot(X, b + k * X, lw=1, c='red',
         label=f'My model: $w_0$ = {b}, $w_1$ = {k}')
axs[0].plot(X, b_sc + k_sc * X, lw=1, c='green',
         label=f'Sclearn: $w_0$ = {b_sc}, $w_1$ = {k_sc}')
axs[0].set_title('Km vs price')
axs[0].set_xlabel('km')
axs[0].set_ylabel('price')

axs[1].plot(n_epoche, score, '-.')
axs[1].set_xlabel('count')
axs[1].set_ylabel('R2')
plt.show()
