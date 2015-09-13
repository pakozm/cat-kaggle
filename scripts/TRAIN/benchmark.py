"""
Author: Francisco Zamora-Martinez
Following script written by Abishek
"""

import datetime
import math
import numpy as np
import pandas as pd
import sklearn as sk

from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error

DATE_DELTA=5
K=10
NUM_TREES=1000
MAX_FEATS="sqrt"
NUM_TREES_CV=100
ONE_HOT_ENCODING=False
CROSS_VALIDATE=True

def split_date(df):
    df["year"]  = df.apply(lambda row: math.floor(row["quote_date"].year/DATE_DELTA)*DATE_DELTA, axis=1)
    df["month"] = df.apply(lambda row: row["quote_date"].month, axis=1)
    df["day"]   = df.apply(lambda row: row["quote_date"].day, axis=1)
    df["dow"]   = df.apply(lambda row: row["quote_date"].dayofweek, axis=1)
    return df.drop("quote_date", axis=1)

def filter_field(df, field, filtered):
    df[field] = df.apply(lambda row: row[field] if row[field] in filtered else float('NaN'), axis=1)

def remove_rare_events(train, test, field, K):
    data = train[field].values
    hist = {}
    for val in data: hist[val] = hist.get(val,0) + 1
    filtered = set( [ key for key,val in hist.items() if val > K ] )
    filter_field(train, field, filtered)
    filter_field(test, field, filtered)

def load_data():
    path = "../input"
    # load training and test datasets
    train = pd.read_csv(path + "/train_set.csv", parse_dates=[2], na_values="NONE")
    test = pd.read_csv(path + "/test_set.csv", parse_dates=[3], na_values="NONE")
    tube = pd.read_csv(path + "/tube.csv", na_values="NONE")
    tube.drop(["num_boss", "num_bracket", "other"], axis=1)
    
    # create some new features
    train = split_date(train)
    test  = split_date(test)

    train = pd.merge(train, tube, on="tube_assembly_id", how="left")
    test  = pd.merge(test, tube, on="tube_assembly_id", how="left")
    
    remove_rare_events(train, test, "supplier", K)
    remove_rare_events(train, test, "material_id", K)
    remove_rare_events(train, test, "end_a", K)
    remove_rare_events(train, test, "end_x", K)

    train_costs = np.log1p( train.cost.values )
    test_ids    = test.id.values.astype(int)

    train = train.drop(["tube_assembly_id", "cost"], axis=1)
    test  = test.drop(["id", "tube_assembly_id"], axis=1)

    train = train.fillna("NA")
    test  = test.fillna("NA")

    for field in ["supplier", "year", "month", "day", "dow",
                  "material_id", "end_a", "end_x",
                  "end_a_1x", "end_a_2x", "end_x_1x",
                  "end_x_2x", "bracket_pricing"]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[field].values))
        train[field] = lbl.transform(train[field].values)
        test[field]  = lbl.transform(test[field].values)

    if ONE_HOT_ENCODING:
        for field in ["supplier", "year", "month", "day", "dow",
                      "material_id", "end_a", "end_x"]:
            train_values = train[field].values
            test_values  = test[field].values
            lbl = preprocessing.OneHotEncoder()
            lbl.fit(np.resize(np.array(train_values).astype(float), (len(train_values),1)))
            train_values = lbl.transform(np.resize(np.array(train_values).astype(float), (len(train_values),1))).toarray()
            test_values  = lbl.transform(np.resize(np.array(test_values).astype(float), (len(test_values),1))).toarray()
            for i in range(train_values.shape[1]):
                train[field + "_" + str(i)] = train_values[:,i]
                test[field + "_" + str(i)]  = test_values[:,i]
            train = train.drop([ field ], axis=1)
            test  = test.drop([ field ], axis=1)

    for field in ["annual_usage", "min_order_quantity", "quantity",
                  "diameter", "wall", "length", "num_bends", "bend_radius"]:
        train[field] = np.log1p( train[field].values )
        test[field]  = np.log1p( test[field].values )
        
    train = np.array(train).astype(float)
    test  = np.array(test).astype(float)
    
    return (train, train_costs, test, test_ids)

train,train_costs,test,test_ids = load_data()
if CROSS_VALIDATE:
    model = ensemble.RandomForestRegressor(n_estimators=NUM_TREES_CV,
                                           max_features=MAX_FEATS,
                                           random_state=123,
                                           verbose=0,
                                           criterion="mse")
    train_shuf,train_costs_shuf = sk.utils.shuffle(train, train_costs,
                                                   random_state=5285)
    scores = cross_val_score(model, train_shuf, train_costs_shuf, cv=10,
                             scoring='mean_squared_error', verbose=100)
    print("CROSS-VALIDATION SCORE:",math.sqrt( abs( scores.mean() ) ))

model = ensemble.RandomForestRegressor(n_estimators=NUM_TREES,
                                       max_features=MAX_FEATS,
                                       random_state=123,
                                       verbose=100,
                                       criterion="mse")
model = model.fit(train, train_costs)
train_p = model.predict(train)
print ("TRAINING SCORE:",np.sqrt(mean_squared_error(train_p, train_costs)))

# get predictions from the model, convert them and dump them!
test_p = np.expm1(model.predict(test))

preds = pd.DataFrame({"cost": test_p}, index = test_ids)
preds.to_csv('benchmark.csv', index=True, index_label="id")
