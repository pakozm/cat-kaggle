



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


if __name__ == '__main__':
    # load training and test datasets
    train = pd.read_csv('../input/team/filledmaterialtrain4.csv' , parse_dates=[2,])#, nrows =1000)
    test = pd.read_csv('../input/team/filledmaterialtest4.csv', parse_dates=[3,] )#, nrows = 1000)
    num_items_train = pd.read_csv('../input/team/number_of_items_per_tube_train.csv')#, nrows = 1000 )
    num_items_test =  pd.read_csv('../input/team/number_of_items_per_tube_test.csv' )#, nrows = 1000 )  
    #freq_compo_train = pd.read_csv('../input/team/freq_compo_bill_train.csv' )#, nrows = 1000 )    
    #freq_compo_test = pd.read_csv('../input/team/freq_compo_bill_test.csv' )#, nrows = 1000 )    
    bill = pd.read_csv('../input/team/bill_connection_type_counts.csv' )#, nrows = 1000 )
    num_spec_train = pd.read_csv('../input/team/num_spec_train.csv')# nrows = 1000 )
    num_spec_test = pd.read_csv('../input/team/num_spec_test.csv')# nrows = 1000 )
    freq_spec_train = pd.read_csv('../input/team/specs_train_merged.csv')
    freq_spec_test = pd.read_csv('../input/team/specs_test_merged.csv')
    #print train.shape , train.columns
    
    
    print '1',train.shape,test.shape
    train = pd.merge(train, bill, on ='tube_assembly_id')
    test = pd.merge(test, bill, on ='tube_assembly_id')
    
    print '2',train.shape,test.shape
    
    #print train.shape , train.columns
    
    for i in range(1,9):
        column_label = 'component_id_'+str(i)
        #print(column_label)
        train[column_label].replace(np.nan,' ', regex=True, inplace= True)
        test[column_label].replace(np.nan,' ', regex=True, inplace= True)
    
    train.fillna(0, inplace = True)
    test.fillna(0, inplace = True)
    
    print '1',train.columns
    
    train['annual_usage'] = 1/(train['annual_usage'].astype(float)+10)
    train['quantity'] = 1/train['quantity'].astype(float)
    
    test['annual_usage'] = 1/(test['annual_usage'].astype(float)+10)
    test['quantity'] = 1/test['quantity'].astype(float)
    

        
    
    train['interactions'] = 0
    test['interactions'] = 0

    
    
    
    
    
    for i in np.unique(test.tube_assembly_id):
       
        test.loc[test.tube_assembly_id == i , 'interactions'] = 1/((test.loc[test.tube_assembly_id == i, 'annual_usage'].astype(float)+10) * test.loc[test.tube_assembly_id == i, 'quantity'].astype(float) )        
        #print  i ,1/((test.loc[test.tube_assembly_id == i, 'annual_usage'].astype(float)+10) * test.loc[test.tube_assembly_id == i, 'quantity'].astype(float) )
        #print train.loc[train.tube_assembly_id == i, 'quantity']



    #print test.loc[test.interactions.isnull(),'tube_assembly_id']

    for i in np.unique(train.tube_assembly_id): 
        train.loc[train.tube_assembly_id == i , 'interactions'] = 1/((train.loc[train.tube_assembly_id == i, 'annual_usage'].astype(float)+10) * train.loc[train.tube_assembly_id == i, 'quantity'].astype(float) )
        #print  i ,1/((train.loc[train.tube_assembly_id == i, 'annual_usage'].astype(float)+10) * train.loc[train.tube_assembly_id == i, 'quantity'].astype(float) )

   

    #fill bend_radius which is equal to 9999.0
    for i in np.unique(train.loc[train.bend_radius == 9999.0 , 'tube_assembly_id']):
        k =  train.loc[(train.tube_assembly_id == i) ,'length'].mean()
        j = train.loc[(train.tube_assembly_id == i) ,'num_bends'].mean()
        #print i,j,k
        #print k ,i , j , train.loc[(train.bend_radius == 9999.0) & (train.tube_assembly_id == i) , 'bend_radius'],'\n'
        train.loc[(train.bend_radius == 9999.0) & (train.tube_assembly_id == i) , 'bend_radius']\
        = train.loc[(train.length > k-5)& (train.num_bends > j-1) & (train.num_bends < j+1) &\
        (train.length < k+5) & (train.bend_radius != 9999.0) , 'bend_radius'].mean()
    
    
    


    # fill length(zero is not possible), admin has given this 
    train.loc[(train.tube_assembly_id == 'TA-00152') & (train.length == 0) , 'length'] = 19
    train.loc[(train.tube_assembly_id == 'TA-00154') & (train.length == 0) , 'length'] = 75
    train.loc[(train.tube_assembly_id == 'TA-00156') & (train.length == 0) , 'length'] = 24
    train.loc[(train.tube_assembly_id == 'TA-01098') & (train.length == 0) , 'length'] = 10
    train.loc[(train.tube_assembly_id == 'TA-01631') & (train.length == 0) , 'length'] = 48
    train.loc[(train.tube_assembly_id == 'TA-03520') & (train.length == 0) , 'length'] = 46
    train.loc[(train.tube_assembly_id == 'TA-04114') & (train.length == 0) , 'length'] = 135
    train.loc[(train.tube_assembly_id == 'TA-17390') & (train.length == 0) , 'length'] = 40
    train.loc[(train.tube_assembly_id == 'TA-18227') & (train.length == 0) , 'length'] = 74
    train.loc[(train.tube_assembly_id == 'TA-18229') & (train.length == 0) , 'length'] = 51
    
    #..................................... for test
    test.loc[(test.tube_assembly_id == 'TA-00152') & (test.length == 0) , 'length'] = 19
    test.loc[(test.tube_assembly_id == 'TA-00154') & (test.length == 0) , 'length'] = 75
    test.loc[(test.tube_assembly_id == 'TA-00156') & (test.length == 0) , 'length'] = 24
    test.loc[(test.tube_assembly_id == 'TA-01098') & (test.length == 0) , 'length'] = 10
    test.loc[(test.tube_assembly_id == 'TA-01631') & (test.length == 0) , 'length'] = 48
    test.loc[(test.tube_assembly_id == 'TA-03520') & (test.length == 0) , 'length'] = 46
    test.loc[(test.tube_assembly_id == 'TA-04114') & (test.length == 0) , 'length'] = 135
    test.loc[(test.tube_assembly_id == 'TA-17390') & (test.length == 0) , 'length'] = 40
    test.loc[(test.tube_assembly_id == 'TA-18227') & (test.length == 0) , 'length'] = 74
    test.loc[(test.tube_assembly_id == 'TA-18229') & (test.length == 0) , 'length'] = 51
    

  


    print '2',train.shape,test.shape

    train['num_comp'] = 0
    train['num_comp'] = num_items_train.row_sums.values  
    test['num_comp']  = 0
    test['num_comp']  = num_items_test.row_sums.values 
    train['num_spec'] = 0    
    train['num_spec'] = num_spec_train.counts.values
    test['num_spec']  = 0
    test['num_spec']  = num_spec_test.counts.values
    
    
   
  
    for i in ['SP.0063','SP.0012','SP.0080','SP.0007','SP.0026','SP.0082','SP.0069','SP.0070']:
       # if i == 'tube_assembly_id':
        #print i            
            #continue
        train[i] = 0
        test[i] = 0
        train[i] = freq_spec_train[i]
        test[i] = freq_spec_test[i]
    
    # ,  'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8',  'component_id_8'
    #'spec1','spec2','spec3','spec4', 'spec5', 'spec8', 'spec9', 'spec10',
    idx = test.id.values.astype(int)
    labels = train.cost.values
    
    train.loc[ train.end_a_1x == 'N' ,'end_a_1x'] = 1
    train.loc[ train.end_a_2x == 'N' , 'end_a_2x'] = 1
    train.loc[ train.end_x_1x == 'N' , 'end_x_1x'] = 1
    train.loc[train.end_x_2x == 'N' ,  'end_x_2x'] = 1

    train.loc[ train.end_a_1x == 'Y' ,'end_a_1x'] = 0
    train.loc[ train.end_a_2x == 'Y' , 'end_a_2x'] = 0
    train.loc[ train.end_x_1x == 'Y' , 'end_x_1x'] = 0
    train.loc[train.end_x_2x == 'Y' ,  'end_x_2x'] = 0



    test.loc[test.end_a_1x == 'N' ,'end_a_1x'] = 1
    test.loc[test.end_a_2x == 'N' , 'end_a_2x'] = 1
    test.loc[test.end_x_1x == 'N' , 'end_x_1x'] = 1
    test.loc[test.end_x_2x == 'N' ,  'end_x_2x'] = 1
    test.loc[test.end_a_1x == 'Y' ,'end_a_1x'] = 0
    test.loc[test.end_a_2x == 'Y' , 'end_a_2x'] = 0
    test.loc[test.end_x_1x == 'Y' , 'end_x_1x'] = 0
    test.loc[test.end_x_2x == 'Y' ,  'end_x_2x'] = 0


    #print test.shape , train.shape
    #print train.columns.get_loc("end_a")
    #print train.columns
    # convert data to numpy array
    
    
    train.loc[train.bracket_pricing == 'Yes', 'bracket_pricing' ] = 1
    train.loc[train.bracket_pricing == 'No', 'bracket_pricing' ] =  0
    
    test.loc[test.bracket_pricing == 'Yes', 'bracket_pricing' ] = 1
    test.loc[test.bracket_pricing == 'No', 'bracket_pricing' ] =  0 
    
    train.loc[train.end_a == 'Yes', 'end_a' ] = 1
    train.loc[train.end_a == 'No', 'end_a' ] = 0
    
    train.loc[train.end_x == 'Yes', 'end_x' ] = 1
    train.loc[train.end_x == 'No', 'end_x' ] = 0
    
    test.loc[test.end_a == 'Yes', 'end_a' ] = 1
    test.loc[test.end_a == 'No', 'end_a' ] = 0
    
    test.loc[test.end_x == 'Yes', 'end_x' ] = 1
    test.loc[test.end_x == 'No', 'end_x' ] = 0
    
    train.to_csv('../input/team/TRAIN.csv')
    test.to_csv('../input/team/TEST.csv')

    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month
    train['dayofyear'] = train.quote_date.dt.dayofyear
    train['dayofweek'] = train.quote_date.dt.dayofweek
    train['day'] = train.quote_date.dt.day
    
    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month
    test['dayofyear'] = test.quote_date.dt.dayofyear
    test['dayofweek'] = test.quote_date.dt.dayofweek
    test['day'] = test.quote_date.dt.day
    
    index = train.columns.get_loc('component_id_1')

    test = test.drop(['supplier','material_id','id' ,  'quote_date' ,'tube_assembly_id',  'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8',  'component_id_8'], axis = 1)
    train = train.drop(['supplier','material_id' ,  'quote_date' ,'tube_assembly_id', 'cost',  'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8',  'component_id_8' ], axis = 1)
   
    print '2',train.columns
    
    train1 = train
    test1 = test
    
    train = np.array(train)
    test = np.array(test)
    
    # object array to float
    train = train.astype(float)
    test = test.astype(float)
    label_log = np.log1p(labels)
    
    
    
  
   
    
    
    
    
    
    
    
    
    
    """
    params = [{'n_estimators': [3500], 'min_samples_split': [35],'max_depth': [8] ,'learning_rate':[0.01] , 'max_features' :['sqrt' ] , 'loss': ['ls']}]
    """    
    gbm1 = ensemble.GradientBoostingRegressor(random_state = 42, n_estimators=3500 , min_samples_split=35, max_depth= 8, learning_rate=.01,max_features = 'sqrt'\
    ,loss ='ls' )
    gbm2 = ensemble.GradientBoostingRegressor(random_state = 42, n_estimators=3500 , min_samples_split=35, max_depth= 10, learning_rate=.01,max_features= 'sqrt'\
    ,loss ='ls' )    
    gbm3 = ensemble.GradientBoostingRegressor(random_state = 42, n_estimators=3500 , min_samples_split=35, max_depth= 12, learning_rate=.01,max_features = 'sqrt'\
    , loss ='ls' ) 
    """
    gbm4 = ensemble.GradientBoostingRegressor(random_state = 42, n_estimators=3500 , min_samples_split=35, max_depth= 9, learning_rate=.01,max_features = 'sqrt'\
    ,loss ='ls' )
    gbm5 = ensemble.GradientBoostingRegressor(random_state = 42, n_estimators=3500 , min_samples_split=35, max_depth= 11, learning_rate=.01,max_features = 'sqrt'\
    ,loss ='ls' )       
    """
    
    
    
   
    # get predictions on test
    print 'fitting started...'
    gbm1.fit(train, label_log)
    gbm2.fit(train, label_log)
    gbm3.fit(train, label_log)    
    #gbm4.fit(train, label_log)
    #gbm5.fit(train, label_log)
    
    
    print 'prediction on test started............'
    # get predictions from the model, convert them and dump them!
    preds1_test = np.expm1(gbm1.predict(test))
    preds2_test = np.expm1(gbm2.predict(test))
    preds3_test = np.expm1(gbm3.predict(test))
    #preds4_test = np.expm1(gbm4.predict(test))
    #preds5_test = np.expm1(gbm5.predict(test))
    
    
    preds1 = (preds1_test+preds2_test+ preds3_test)/3#+preds4_test+preds5_test)/5.0   
    preds1 = pd.DataFrame({"id": idx, "cost": preds1})
    preds1.to_csv('81206.csv', index=False)
    

    """
    print 'taking the ranking avg  of first three models and then save the results'
    preds1 = (preds1_test+preds2_test+ preds3_test)/3#+preds4_test+preds5_test)/5.0   
    preds1 = pd.DataFrame({"id": idx, "cost": preds1})
    preds1.to_csv('81206.csv', index=False)
    
    
    

    #print 'kk'
    #clf = grid_search.GridSearchCV(gbm,params, verbose=1,  n_jobs = 2)
    #label_log = np.log1p(labels)
    # cross validation
    #print("k-Fold RMSLE:")
    #clf = grid_search.GridSearchCV(gbm, params, verbose=1 , n_jobs = 2)
    cv_rmsle1 = cross_validation.cross_val_score(gbm1, train,  label_log,cv = 10, scoring='mean_squared_error')
    print '1.... score',cv_rmsle1
    cv_rmsle1 = np.sqrt(np.abs(cv_rmsle1))
    print 'mean...', np.mean(cv_rmsle1)
    #print(cv_rmsle), "asdasdad"
    #print("Mean: " + str(cv_rmsle.mean()))
    
    cv_rmsle2 = cross_validation.cross_val_score(gbm2, train,  label_log, cv = 10,scoring='mean_squared_error')
    print '2.......',cv_rmsle2
    cv_rmsle2 = np.sqrt(np.abs(cv_rmsle2))
    print 'mean...', np.mean(cv_rmsle2) 
    
    cv_rmsle3 = cross_validation.cross_val_score(gbm3, train, label_log, cv = 10, scoring='mean_squared_error')
    print '2.......',cv_rmsle3
    cv_rmsle3 = np.sqrt(np.abs(cv_rmsle3))
    print 'mean...', np.mean(cv_rmsle3)  
   
    print 'prediction  on training itself started'
    preds1_train = np.expm1(gbm1.predict(train))
    preds2_train = np.expm1(gbm2.predict(train))    
    preds3_train = np.expm1(gbm3.predict(train))
    #preds4_train = np.expm1(gbm4.predict(train))
    #preds5_train = np.expm1(gbm5.predict(train))  
    
    print 'make train test prediction as an array for make use of svm Regressor'
    preds_train = np.transpose(np.vstack((preds1_train,preds2_train,preds3_train)))   
    preds_test = np.transpose(np.vstack((preds1_test,preds2_test,preds3_test)))
    
    
    
    
    
    print "Blending."
    clf = LogisticRegression()
    clf.fit(preds_train, labels)
    blend_pred = clf.predict(preds_test)
    blend_pred = pd.DataFrame({"id": idx, "cost": blend_pred})
    preds1.to_csv('logistic_reg.csv', index=False)
    
    
    
   
    clasfr = svm.SVR(random_state= 42)
    
    param_grid = [{'C':[5,10] , 'gamma':[0.0,0.01],'degree':[1,2,3], 'kernel':['linear', 'rbf'], 'tol':[0.001,0.002]}]  
    clf = grid_search.GridSearchCV(clasfr, param_grid, verbose=1 , n_jobs = 2)
    
    print 'grid search started'
    cv_rmsle = cross_validation.cross_val_score(clf, preds_train, labels, scoring='mean_squared_error')
    print(cv_rmsle)
    cv_rmsle = np.sqrt(np.abs(cv_rmsle))
    print(cv_rmsle), "asdasdad"
    print("Mean: " + str(cv_rmsle.mean()))
    
   
    
    clf.fit(preds_train,labels)
    preds_svm =clf.predict(preds_test)
    #preds = (0.7*preds1_test+ 0.15*preds2_test+ 0.15*preds3_test)    
    preds = pd.DataFrame({"id": idx, "cost": preds_svm})
    preds.to_csv('blending_with_svm.csv', index=False)
    
    
   
    np.random.seed(0)
    n_folds = 10
    verbose = True
    X, y, X_submission = train,label_log,test
    
    
    
        
        
        
        
        
        
    skf = list(StratifiedKFold(train1.bracket_pricing, n_folds))
    
    clfs = [RandomForestRegressor(n_estimators=3000,max_depth=10,max_features= 'sqrt',min_samples_split= 35,n_jobs=2,random_state = 42),
            RandomForestRegressor(n_estimators=3000,max_depth=10,max_features= 'log2',min_samples_split= 35,n_jobs=2,random_state = 42),
            ExtraTreesRegressor(n_estimators=3000,max_depth=10,max_features= 'sqrt',min_samples_split= 35,n_jobs=2,random_state = 42),
            GradientBoostingRegressor(random_state = 42, loss= 'ls', learning_rate=0.01, n_estimators= 3500, max_features= 'sqrt', min_samples_split= 35, max_depth= 8),
            GradientBoostingRegressor(random_state = 42, loss= 'ls', learning_rate=0.01, n_estimators= 3500, max_features= 'sqrt', min_samples_split= 35, max_depth= 10),
            GradientBoostingRegressor(random_state = 42, loss= 'ls', learning_rate=0.01, n_estimators= 3500, max_features= 'log2', min_samples_split= 35, max_depth= 12)]
    
    
    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train1, test1) in enumerate(skf):
            print "Fold", i
            X_train = X[train1]
            y_train = y[train1]
            X_test = X[test1]
            y_test = y[test1]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test1, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    
    
    
    
    
    
    
    
    print
    print "Blending."
    clf = LogisticRegression(penalty = 'l1', c= .01, max_iter = 100 )
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)

    print "Linear stretch of predictions to [0,1]"
    preds = np.expm1(y_submission)

    
    preds = pd.DataFrame({"id": idx, "cost": preds})
    preds.to_csv('gradboost_changed_name_blend.csv', index=False)  
    
    
    
    dataset_blend_test1 = np.expm1(dataset_blend_test)
    preds = .05*dataset_blend_test1[:,0] + .05*dataset_blend_test1[:,1] + .05*dataset_blend_test1[:,2]+.05*dataset_blend_test1[:,3] + .7*dataset_blend_test1[:,4] +.1*dataset_blend_test1[:,5]   
    preds = pd.DataFrame({"id": idx, "cost": preds})
    preds.to_csv('gradboost_changed_name_blend2.csv', index=False)
    
    
    
    
    
    
    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
  

    
    gbm = ensemble.GradientBoostingRegressor(random_state=42)

    params = [{'n_estimators': [3000,3500], 'min_samples_split': [30,35],'max_depth': [10,11,12], 'min_samples_leaf':[1,2,3],'subsample': [0.9,1.0], \
    'learning_rate':[.009,0.01,.011] , 'max_features' :['sqrt', 'log2' ] , 'loss': ['ls']}   ]
              
    params = [{'n_estimators': [2500, 3000], 'min_samples_split': [25,30],'max_depth': [12,13], 'min_samples_leaf':[1,2,3],\
    'learning_rate':[0.01] , 'max_features' :['sqrt'] , 'loss': ['ls']}]
    print 'kk'
    clf = grid_search.GridSearchCV(gbm,params, verbose=1 , n_jobs = 2)
    label_log = np.log1p(labels)
    # cross validation
    print("k-Fold RMSLE:")
    cv_rmsle = cross_validation.cross_val_score(clf, train, label_log, scoring='mean_squared_error')
    print(cv_rmsle)
    cv_rmsle = np.sqrt(np.abs(cv_rmsle))
    print(cv_rmsle), "asdasdad"
    print("Mean: " + str(cv_rmsle.mean()))
    
    # get predictions on test
    clf.fit(train, label_log)
    
    # get predictions from the model, convert them and dump them!
    preds = np.expm1(clf.predict(test))
    preds = pd.DataFrame({"id": idx, "cost": preds})
    preds.to_csv('gradboost_changed_name.csv', index=False)






label_log = np.log1p(labels)

# fit a gbm model
gbm = ensemble.GradientBoostingRegressor(random_state=42)
rf = ensemble.RandomForestRegressor()

# tune parameters
parameters = {"max_depth": [3, None] ,'n_estimators':(500,1000,1500,2000),'max_features' :['sqrt']}
clf = grid_search.GridSearchCV(rf, parameters, verbose=1)

# cross validation
print("k-Fold RMSLE:")
cv_rmsle = cross_validation.cross_val_score(clf, train, label_log, scoring='mean_squared_error')
print(cv_rmsle)
cv_rmsle = np.sqrt(np.abs(cv_rmsle))
print(cv_rmsle)
print("Mean: " + str(cv_rmsle.mean()))

# get predictions on test
clf.fit(train, label_log)

# get predictions from the model, convert them and dump them!
preds = np.expm1(clf.predict(test))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('benchmark.csv', index=False)


print 'just testing my score agauist  grid search'
    params = [{'n_estimators': [3000,3500], 'min_samples_split': [30,35],'max_depth': [8,10] ,\
    'learning_rate':[0.01,0.009] , 'max_features' :['sqrt' ] , 'loss': ['ls']}]
    gbm = ensemble.GradientBoostingRegressor(random_state = 42)
    print 'kk'
    clf = grid_search.GridSearchCV(gbm,params, verbose=1,  n_jobs = 2)
    label_log = np.log1p(labels)
    # cross validation
    print("k-Fold RMSLE:")
    #clf = grid_search.GridSearchCV(gbm, params, verbose=1 , n_jobs = 2)
    cv_rmsle = cross_validation.cross_val_score(clf, train, label_log, scoring='mean_squared_error')
    print(cv_rmsle)
    cv_rmsle = np.sqrt(np.abs(cv_rmsle))
    print(cv_rmsle), "asdasdad"
    print("Mean: " + str(cv_rmsle.mean()))
 # get predictions on test
    clf.fit(train, label_log)
    
    # get predictions from the model, convert them and dump them!
    preds = np.expm1(clf.predict(test))
    preds = pd.DataFrame({"id": idx, "cost": preds})
    preds.to_csv('After_Adding_Vol.csv', index=False)


"""





