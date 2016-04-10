import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import accuracy_score

import random

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

if len(train_df.Fare[ train_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = train_df[ train_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        train_df.loc[ (train_df.Fare.isnull()) & (train_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

#drop col
drop_col = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'Sex', 'Embarked']
predictors = ["Survived", "Pclass", "Gender", "Age", "SibSp", "Parch", "Embarked"]
predictors_t = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Embarked"]

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(drop_col, axis=1) 
#train_df = train_df[predictors]

# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(drop_col, axis=1) 
#test_df = test_df[predictors_t]

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


idx = range(len(train_data))
random.shuffle(idx)
train_idx = idx[:662]
test_idx = idx[662:]


train_X = train_data[0::,1::]
train_y = train_data[0::,0]

train_X_tr = [train_X[i] for i in train_idx]
train_y_tr = [train_y[i] for i in train_idx]
train_X_te = [train_X[i] for i in test_idx]
train_y_te = [train_y[i] for i in test_idx]


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

        forest = RandomForestClassifier(n_estimators=x, max_depth=y)
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
        print x, accuracy_score(y_true, y_pred) 
        
        res.append((x, accuracy_score(y_true, y_pred), 'xgboost', str(y)))

res = sorted(res, key = lambda x:x[1], reverse = True)
print res
cls = {'gbdt':GradientBoostingClassifier, 'rf':RandomForestClassifier, 'xgboost':xgb.XGBClassifier}
n = int(res[0][0])
max_depth = int(res[0][3])
cla = cls[res[0][2]] 
print n, cla, max_depth
fn = '_'.join([str(n), str(res[0][2]), str(max_depth)])


#forest = GradientBoostingClassifier(n_estimators=150, max_depth=3)
forest = cla(n_estimators=n, max_depth=max_depth)

forest = forest.fit( train_X, train_y )

output = forest.predict(test_data).astype(int)

predictions_file = open("zx_cross_"+fn+".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()