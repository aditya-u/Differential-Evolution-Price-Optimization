import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import yabox
import pickle
import numpy as np
import sys

def return_inverse_profit(cost, comp1, comp2, comp3, item):
    def f(x):
        global xgb
        global df1
        inp = np.array([cost, float(x), comp1, comp2, comp3]).reshape(1,-1)
        a = xgb.predict(inp)
        dfx = df1[df1.SCRUB_ITEM == item]
        scaler = MinMaxScaler()
        scaler.fit(dfx['unit_qty'].values.reshape(-1,1))
        a = scaler.inverse_transform(np.array(a).reshape(1,-1))
        scaler.fit(dfx['unit_cost'].values.reshape(-1,1))
        unit_cost = scaler.inverse_transform(np.array(cost).reshape(1,-1))
        scaler.fit(dfx['unit_price'].values.reshape(-1,1))
        x = scaler.inverse_transform(np.array(x).reshape(1,-1))
        profit = (a * (x - unit_cost))
        inverse_profit = 10000000-profit
        return inverse_profit
    return f

def dynamic_pricing(item, comp1, comp2, comp3, cost):
    global df1    
    scaler = MinMaxScaler()
    scaler = scaler.fit(df1[df1.SCRUB_ITEM == item].comp1_price.values.reshape(-1,1))
    comp1 = float(scaler.transform(np.array(comp1).reshape(1,-1))[0])
    scaler = scaler.fit(df1[df1.SCRUB_ITEM == item].comp2_price.values.reshape(-1,1))
    comp2 = float(scaler.transform(np.array(comp2).reshape(1,-1))[0])
    scaler = scaler.fit(df1[df1.SCRUB_ITEM == item].comp3_price.values.reshape(-1,1))
    comp3 = float(scaler.transform(np.array(comp3).reshape(1,-1))[0])
    scaler = scaler.fit(df1[df1.SCRUB_ITEM == item].unit_cost.values.reshape(-1,1))
    cost = float(scaler.transform(np.array(cost).reshape(1,-1))[0])
    f = return_inverse_profit(cost, comp1, comp2, comp3, item)
    out = yabox.DE(f, [(0.00,1.00)], maxiters=1000)
    x0, y0 = out.solve(show_progress=True)
    dfx = df1[df1.SCRUB_ITEM == item]
    scaler = MinMaxScaler()
    scaler.fit(dfx['unit_price'].values.reshape(-1,1))
    x0 = scaler.inverse_transform(np.array(x0).reshape(1,-1))
    y0 = 10000000-y0
    return (x0[0][0])

xgb = pickle.load(open("model.pickle.dat", "rb"))
df1 = pd.read_csv('TSC_PRICE_OPT.csv')
items = df1.SCRUB_ITEM.unique()
df2 = pd.DataFrame(columns = ['unit_cost','unit_price','comp1_price','comp2_price','comp3_price','unit_qty'])
for i in items:
    x = df1[df1.SCRUB_ITEM == i]
    x = x.drop('SCRUB_ITEM', axis=1)
    scaler = MinMaxScaler()
    df2 = pd.concat([df2,pd.DataFrame(scaler.fit_transform(x.to_numpy()),columns = ['unit_cost','unit_price','comp1_price','comp2_price','comp3_price','unit_qty'])])
df2=df2.reset_index(drop=True)

# print(dynamic_pricing(1, 205.58, 205.58, 9999999999, 132.97))
item = float(sys.argv[1])
comp1 = float(sys.argv[2])
comp2 = float(sys.argv[3])
comp3 = float(sys.argv[4])
cost = float(sys.argv[5])
optim_price = dynamic_pricing(item, comp1, comp2, comp3, cost)
print("Item number: ", item)
print('Price of Competitor 1', comp1)
print('Price of Competitor 2', comp2)
print('Price of Competitor 3', comp3)
print("Cost of production: ", cost)
print("Optimal price: ", optim_price)

