#!/usr/bin/env python3
from libs import snackbar
import pandas as pd

dt = pd.DataFrame();
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

check = False
while check == False :
    mileage = input("Enter km: ")
    try :
        number_mileage = float(mileage) - 0
        if (number_mileage >= 0) :
            check = True
        else :
            snackbar("Error: negative mileage? Try again.", 'info')
    except :
        snackbar("Error: not a number, try again.", 'info')

snackbar(float(dt['b'][0]) + (float(dt['k'][0]) * number_mileage), 'success')
