from glob import glob
from sklearn import ensemble, preprocessing, grid_search, cross_validation
import pandas as pd
import numpy as np
from sklearn import ensemble , preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train_set.csv")
n_folds = 5
cv_generator = StratifiedKFold(train.bracket_pricing, n_folds, shuffle=True, random_state=425)

val_list  = glob("val_stage0_*")
test_list = glob("test_stage0_*")
val_list.sort()
test_list.sort()
train_matrix = np.log1p( np.concatenate([ np.array(pd.read_csv(x)["cost"])[:,np.newaxis] for x in val_list ], axis=1) )
test_matrix = np.log1p( np.concatenate([ np.array(pd.read_csv(x)["cost"])[:,np.newaxis] for x in test_list ], axis=1) )
label_log = np.log1p(np.array(train["cost"]))

print(preds_val)
print(preds_test)

nn = 
log_preds_val = cross_validation.cross_val_predict(, train,
                                                   label_log,
                                                   cv=cv_generator,
                                                   n_jobs=4,
                                                   verbose=1)
