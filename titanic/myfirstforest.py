import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score

import random
train_df = pd.read_csv('train.csv', header=0)    


train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

if len(train_df.Fare[ train_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = train_df[ train_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        train_df.loc[ (train_df.Fare.isnull()) & (train_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


drop_col = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'Sex']
#predictors = ["Survived", "Pclass", "Gender", "Age", "SibSp", "Parch", "Embarked"]
#predictors_t = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Embarked"]

train_df = train_df.drop(drop_col, axis=1) 
#train_df = train_df[predictors]

test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe


test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age


if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


ids = test_df['PassengerId'].values

test_df = test_df.drop(drop_col, axis=1) 
#test_df = test_df[predictors_t]


train_data = train_df.values
test_data = test_df.values

idx = range(len(train_data))
random.shuffle(idx)
train_idx = idx[:662]
test_idx = idx[662:]


train_X = train_data[0::,1::]
train_y = train_data[0::,0]

"""
train_X_tr = [train_X[i] for i in train_idx]
train_y_tr = [train_y[i] for i in train_idx]
train_X_te = [train_X[i] for i in test_idx]
train_y_te = [train_y[i] for i in test_idx]


RandomForestClassifier = ExtraTreesClassifier

res = []
for x in xrange(50, 1000, 50):
    for y in ([3]):
        forest = GradientBoostingClassifier(n_estimators=x, max_depth=y)
        forest = forest.fit( train_X_tr, train_y_tr )   
        output = forest.predict(train_X_te).astype(int)
        y_true = train_y_te
        y_pred = output 
        print x, accuracy_score(y_true, y_pred) 

        res.append((x, accuracy_score(y_true, y_pred), 'gbdt', str(y))) 

        forest = RandomForestClassifier(n_estimators=x, max_depth=y, max_features= 3, criterion= 'entropy',min_samples_split= 1, min_samples_leaf= 2, n_jobs = -1)
        forest = forest.fit( train_X_tr, train_y_tr )   

        output = forest.predict(train_X_te).astype(int)
        y_true = train_y_te
        y_pred = output 
        print x, accuracy_score(y_true, y_pred) 

        res.append((x, accuracy_score(y_true, y_pred), 'rf', str(y)))

        forest = xgb.XGBClassifier(max_depth=y, n_estimators=x, learning_rate=0.05)
        forest = forest.fit( train_X_tr, train_y_tr )   
        output = forest.predict(train_X_te).astype(int)
        y_true = train_y_te
        y_pred = output 
        print x, accuracy_score(y_true, y_pred)  k
        
        res.append((x, accuracy_score(y_true, y_pred), 'xgboost', str(y)))

res = sorted(res, key = lambda x:x[1], reverse = True)
print res
cls = {'gbdt':GradientBoostingClassifier, 'rf':RandomForestClassifier, 'xgboost':xgb.XGBClassifier}
n = int(res[0][0])
max_depth = int(res[0][3])
cla = cls[res[0][2]] 
print n, cla, max_depth
fn = '_'.join([str(n), str(res[0][2]), str(max_depth)])
"""
"""
n = 2000
max_depth = 3
"""
fn = 'tmp'
cla = ExtraTreesClassifier

#forest = GradientBoostingClassifier(n_estimators=150, max_depth=3)
tree = cla(n_estimators=1000, max_features= 3,criterion= 'entropy',min_samples_split= 1, max_depth=20 , min_samples_leaf= 2, n_jobs = -1)
#forest = BaggingRegressor(tree, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

forest = tree.fit( train_X, train_y )

output = forest.predict(test_data).astype(int)

predictions_file = open("zx_cross_"+fn+".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()