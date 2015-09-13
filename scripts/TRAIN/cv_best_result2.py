NCORES=2

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

class LuaBlocksCV(cross_validation._PartitionIterator):
    
    def __init__(self, blocks):
        super(LuaBlocksCV, self).__init__(sum(map(len,blocks)))
        self.blocks = blocks
        
    def _iter_test_indices(self):
        for i in range(len(self.blocks)):
            # it comes from Lua, so we need to substract one before yielding
            yield np.array(self.blocks[i]) - 1    

np.random.seed(0)

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
    
    #train.to_csv('../input/team/TRAIN.csv')
    #test.to_csv('../input/team/TEST.csv')

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

    test = test.drop(['dayofyear', 'supplier','material_id','id' ,  'quote_date' ,'tube_assembly_id',  'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8',  'component_id_8'], axis = 1)
    train = train.drop(['dayofyear', 'supplier','material_id' ,  'quote_date' ,'tube_assembly_id', 'cost',  'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8',  'component_id_8' ], axis = 1)
   
    print '2',train.columns
    
    train1 = train
    test1 = test
    
    train = np.array(train)
    test = np.array(test)
    
    # object array to float
    train = train.astype(float)
    test = test.astype(float)
    label_log = np.log1p(labels)
    
    
    
  
   
    
    
    
    
    blocks = [ map(int, line.rstrip().split()) for line in open("BLOCKS.txt").readlines() ]

    """
    params = [{'n_estimators': [3500], 'min_samples_split': [35],'max_depth': [8] ,'learning_rate':[0.01] , 'max_features' :['sqrt' ] , 'loss': ['ls']}]
    """

    def generate_models():
        return [
            RandomForestRegressor(n_estimators=4300,max_depth=8,max_features= 13,min_samples_split= 33,random_state = 42),
            RandomForestRegressor(n_estimators=4300,max_depth=10,max_features= 13,min_samples_split= 33,random_state = 2),
            ExtraTreesRegressor(n_estimators=4300,max_depth=8,max_features= 13,min_samples_split= 33,random_state = 32),
            GradientBoostingRegressor(random_state = 1, loss= 'ls', learning_rate=0.01, n_estimators= 4300, max_features= 13, min_samples_split= 33, max_depth= 8),
            GradientBoostingRegressor(random_state = 4, loss= 'ls', learning_rate=0.01, n_estimators= 4300, max_features= 13, min_samples_split= 33, max_depth= 10),
            GradientBoostingRegressor(random_state = 100, loss= 'ls', learning_rate=0.01, n_estimators= 4300, max_features= 13, min_samples_split= 33, max_depth= 12),
            GradientBoostingRegressor(random_state = 256, loss= 'ls', learning_rate=0.01, n_estimators= 4300, max_features= 13, min_samples_split= 33, max_depth= 11),
            GradientBoostingRegressor(random_state = 37, loss= 'ls', learning_rate=0.01, n_estimators= 4300, max_features= 13, min_samples_split= 33, max_depth= 9)
        ]
    
    def loss(log_p, log_t):
        return np.sqrt(np.mean((log_p - log_t)**2))
    
    models = generate_models()

    # get predictions on training (using CV) NEEDS SKLEARN >= 0.16
    print 'cross validation started...'
    for i in range(len(models)):
        log_preds_val = cross_validation.cross_val_predict(models[i], train,
                                                           label_log,
                                                           cv=LuaBlocksCV(blocks),
                                                           n_jobs=NCORES,
                                                           verbose=1)
        print i,loss(log_preds_val, label_log)
        pd.DataFrame({"cost": np.expm1(log_preds_val)}).to_csv('val_stage0_%d.csv'%(i), index=True)

    models = generate_models()
    
    # get predictions on test
    print 'fitting with all training started...'
    for gbm in models:
        gbm.fit(train, label_log)
    
    print 'prediction on test started............'
    # get predictions from the model, convert them and dump them!
    preds_test = [ np.expm1(gbm.predict(test)) for gbm in models ]
    
    #preds1 = (preds1_test+preds2_test+ preds3_test)/3#+preds4_test+preds5_test)/5.0
    for i in range(len(preds_test)):
        pd.DataFrame({"id": idx, "cost": preds_test[i]}).to_csv('test_stage0_%d.csv'%(i), index=False)
