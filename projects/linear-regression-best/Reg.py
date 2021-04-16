from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import sys

## initialize constants and filename
TRAINDATA = sys.argv[1]
TESTDATA = sys.argv[2]

# read in the data from the file in the current directory
# not currently configured to read in from command line
df_train = pd.read_csv(TRAINDATA, sep="\t")
df_test = pd.read_csv(TESTDATA, sep="\t")

# separate input features and target
# split testing and training sets

x_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]

x_test = df_test

def rss_score(y, y_pred):
    return np.sum((y - y_pred)**2)

def baseline():
    # define model
    model = LinearRegression().fit(x_train, y_train)

    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    cvScores = cross_val_score(model,
                                x_train, y_train,
                                scoring='r2',
                                cv=cv,
                                n_jobs=-1)
    
    # report performance
    print('CV Mean: ', np.mean(cvScores))
    print('STD: ', np.std(cvScores))
    y_pred = model.predict(x_train)
    print('RSS: %s' % rss_score(y_train, y_pred))
    print('R2: %.3f +/- (%.3f)' % (np.mean(cvScores), np.std(cvScores)))

def get_cv_ridge():
    # define model
    model = Ridge()
    
    # define evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # define search space
    space = dict()
    space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
    space['alpha'] = np.arange(0.010, 0.020, 0.001)
    space['fit_intercept'] = [True, False]
    space['normalize'] = [True, False]

    # define search
    search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
    print("Performing grid search CV...")

    # result
    result = search.fit(x_train,y_train)

    # get best parameters
    best = result.best_params_
    print("Best Hyperparameters: %s" % best)

    # final model
    final_model = Ridge().set_params(**best)

    # fit the final model
    final_model.fit(x_train, y_train)

    # find r2 and rss
    y_pred = final_model.predict(x_train)
    print("Final Model RSS: %s" % rss_score(y_train, y_pred))
    print("Final Model R2: %s" % r2_score(y_train, y_pred))

    # predict the output of y
    predictions = final_model.predict(x_test)
    
    # write predictions to file
    # with open('predictions.txt', 'a') as f:
    #     for p in predictions:
    #         f.write(str(p)+"\n")

print("Performing baseline model...")
baseline()
print("---------FINISHED BASELINE MODEL------------")
print("---------STARTING PERFORMANCE MODEL------------")
get_cv_ridge()
