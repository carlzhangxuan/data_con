import random
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

def data_loader(fn):
    return pd.read_csv(fn)

def check_col_type(data_f, target):
    for col in data_f.columns:
        if data_f[col].dtype == 'O':    
            yield ('cat', col, check_col_hist(target, data_f[col]))
        else:
            yield ('con', col, check_col_cont(target, data_f[col]))

def check_col_hist(target, col):
    t_dic = {}
    for t, v in zip(target, col):
        if t not in t_dic:
            t_dic[t] = {}
        if v not in t_dic[t]:
            t_dic[t][v] = 1
        t_dic[t][v] += 1
    return t_dic

def check_col_cont(target, col, mean_for_missing=True):
    t_dic = {}
    res = {}
    missing_data = len(col[col.isnull()])
    if mean_for_missing:
        if missing_data > 0:
            #col[col.isnull()] = col.mean()
            col[col.isnull()] = -10
            print col[col.isnull()]
    for t, v in zip(target, col):
        if t not in t_dic:
            t_dic[t] = []
        t_dic[t].append(v)
    for ki in t_dic:
        res[ki] = sorted(t_dic[ki])
    return res

def prepare(num=2): 
    n = 0
    n_bins = 50
    train_fn = 'data/train.csv'
    train_data = data_loader(train_fn)
    target = train_data['target']
    iterer = check_col_type(train_data, target)
    iterer.next()
    while n < num:
        n += 1
        flag, k, v = iterer.next()
        if flag == 'con':
            norm = max(np.array(v[0]).max(), np.array(v[1]).max())
            print norm
            plt.subplot(121)
            plt.title(k+' 0')
            plt.hist(np.array(v[0])/norm, n_bins)
            plt.subplot(122)
            plt.title(k+' 1')
            plt.hist(np.array(v[1])/norm, n_bins)
            plt.show()

def data_prepare(f_k=20):
    train_fn = 'data/train.csv'
    test_fn = 'data/test.csv'
    train = data_loader(train_fn)
    test = data_loader(test_fn)

    y = train['target'].values
    t_id = test['ID'].values
    train = train.drop(['ID', 'target'], axis = 1)
    test = test.drop(['ID'], axis = 1)

    #drop col with missing values
    #train = train[['v3','v10','v12','v14','v21','v22','v24','v30','v31','v34','v38','v40','v47','v50','v52','v56','v62','v66','v71','v72','v74','v75','v79','v91','v107','v110','v112','v113','v114','v125','v129']]
    #test = test[['v3','v10','v12','v14','v21','v22','v24','v30','v31','v34','v38','v40','v47','v50','v52','v56','v62','v66','v71','v72','v74','v75','v79','v91','v107','v110','v112','v113','v114','v125','v129']]
    
 
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
        if train_series.dtype == 'O':
            train[train_name], tmp_indexer = pd.factorize(train[train_name])
            test[test_name] = tmp_indexer.get_indexer(test[test_name])
        else:
            tmp_len = len(train[train_series.isnull()])
            if tmp_len>0:
                #train.loc[train_series.isnull(), train_name] = train_series.mean()
                train.loc[train_series.isnull(), train_name] = train_series.min()*2

            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                #test.loc[test_series.isnull(), test_name] = test_series.mean()
                test.loc[test_series.isnull(), test_name] = test_series.min()*2


    #feature selection by ANOVA 
    """
    feature_s = SelectKBest(f_classif, k=f_k).fit(train, y)
    f_index = feature_s.get_support()
    feature_sd = [x[0] for x in  filter(lambda (x, y): y == True, zip(train.columns, f_index))]
    print feature_sd
    train = train[feature_sd]
    test = test[feature_sd]
    """
    
    X = train.values
    t = test.values
    sample_len = len(y)
    idx = range(sample_len)
    random.shuffle(idx)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    return (y, X, t_id, t)
    
def train(f_k=50, n_folds=10, KFold=False, max_depth=3, n_estimators=200, learning_rate=0.05):
    y, X, t_id, t = data_prepare(f_k)
    sample_len = len(y)
    if KFold:
        kf = KFold(sample_len, n_folds=n_folds)
        for train_idx, valid_idx in kf:
            train_X_tr = [X[i] for i in train_idx]
            train_y_tr = [y[i] for i in train_idx]
            train_X_va = [X[i] for i in valid_idx]
            train_y_va = [y[i] for i in valid_idx]

            #forest = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
            forest = ExtraTreesClassifier(n_estimators=700, max_features= 50,criterion= 'entropy', min_samples_split= 5, max_depth= 50, min_samples_leaf= 5)
            forest = forest.fit(train_X_tr, train_y_tr)
            output = forest.predict(train_X_va).astype(int)
            print accuracy_score(train_y_va, output)
    else:
        idx = range(sample_len)
        th = int(math.ceil(sample_len*0.7))
        train_idx = idx[:th]
        valid_idx = idx[th:]
        train_X_tr = [X[i] for i in train_idx]
        train_y_tr = [y[i] for i in train_idx]
        train_X_va = [X[i] for i in valid_idx]
        train_y_va = [y[i] for i in valid_idx]
    
    #forest = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
    forest = ExtraTreesClassifier(n_estimators=700, max_features= 50,criterion= 'entropy', min_samples_split= 5, max_depth= 50, min_samples_leaf= 5)
    forest = forest.fit(train_X_tr, train_y_tr)
    output = forest.predict(train_X_va).astype(int)
    print accuracy_score(train_y_va, output)
    print log_loss(train_y_va, output)

    return (forest, t_id, t)

def predict(f_k=50, trained=False):
    print 'ID,PredictedProb'
    if trained:
        model, tid, test = train()
        output = model.predict_proba(test)

    else:
        print 'train again'
        y, X, tid, test = data_prepare(f_k=f_k)
        #forest = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
        forest = ExtraTreesClassifier(n_estimators=1000,max_features= 50,criterion= 'entropy',min_samples_split= 4, max_depth= 35, min_samples_leaf= 2, n_jobs = -1)
        forest = forest.fit(X, y)
        output = forest.predict_proba(test)
    
    for (ids, pre) in zip(tid, output):
        print ','.join([str(ids), str(pre[1])])
  
if __name__ == '__main__':
    import time
    st = time.time()
    flg = False
    #prepare(100)
    #data_prepare()
    #train(f_k=50)
    predict(f_k=51, trained=flg)
    ed = time.time()
    print int(ed - st)








