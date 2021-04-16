from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


import numpy as np
import pandas as pd
import sys

## initialize constants and filename
FILE_NAME = sys.argv[1]
TRAIN_DATA = sys.arg[2]
TEST_DATA = sys.arg[3]

# read in the data from the file in the current directory
# not currently configured to read in from command line
df = pd.read_csv(FILE_NAME, sep="\t")
df_train = pd.read_csv(TRAIN_DATA, sep="\t")

# separate input features and target
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# split testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=27)

## model
# get cv scores 


def get_cv_scores(model):
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    cvScores = cross_val_score(model,
                                X_test, y_test,
                                scoring='r2',
                                cv=cv,
                                n_jobs=-1)

    
    # report performance
    print('CV Mean: ', np.mean(cvScores))
    print('STD: ', np.std(cvScores))
    y_pred = model.predict(X_test)
    rss = (np.array(y_test) - np.array(y_pred)).sum()
    print('RSS: {}'.format(rss))
    print('R2: %.3f +/- (%.3f)' % (np.mean(cvScores), np.std(cvScores)))
    headers =   ['Fixed Acidity', 
                'Volatile Acidity', 
                'Citric Acid', 
                'Residual Sugar', 
                'Chlorides', 
                'Free Sulfur Dioxide', 
                'Total Sulfur Dioxide', 
                'Density', 
                'pH', 
                'Sulfates', 
                'Alcohol']
    print('------ Coefficients Below ------')
    for i, coef in enumerate(model.coef_):
        header = headers[i]
        print(header, ':', coef)
        
        

# get cv scores for Linear Regression

def get_cv_lr():
    # Train model
    lr = LinearRegression().fit(X_train, y_train)
    # get cross val scores
    get_cv_scores(lr)

get_cv_lr()

