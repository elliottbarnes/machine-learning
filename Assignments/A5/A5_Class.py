# basic
import sys
import numpy as np
import pandas as pd
# graphing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve

# read in file names from the command line
TRAIN_DATA = sys.argv[1]
TEST_DATA = sys.argv[2]

# create training and test data frames 
df_train    = pd.read_csv(TRAIN_DATA, sep="\t")
df_test     = pd.read_csv(TEST_DATA, sep="\t")

## clean data
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)
# seperate test and train input/output features
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]

X_test  = df_test

# set up classifiers to use
classifier_dict = {
    # "knn": {
    #     "clf": KNeighborsClassifier(),
    #     "params": {
    #         'n_neighbors': [5, 11, 19, 29],
    #         'weights': ['uniform', 'distance'],
    #         'metric': ['euclidean', 'manhattan']
    #     },
    #     "best": None,
    #     "pr_score": None
    # },
    "Ada Boost": {
        "clf": AdaBoostClassifier(),
        "params": {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.001, 0.01, 0.1],
            'algorithm': ["SAMME", "SAMME.R"]
        },
        "best": None,
        "pr_score": None
    },
    "SVC": {
        "clf": SVC(),
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ['rbf', 'sigmoid'],
            "gamma": [0.1, 1, 10, 100]
        },
        "best": None,
        "pr_score": None
    }
}

# find best hyperparameter setting for each classifier
for key, value in classifier_dict.items():
    gs = GridSearchCV(
        estimator=value["clf"],
        param_grid=value["params"],
        scoring="f1",
        n_jobs=-1,
    )
    gs_r = gs.fit(X_train, y_train)
    classifier_dict[key]["best"] = gs_r.best_params_


for key, value in classifier_dict.items():
    print("Best params: ", classifier_dict[key]["best"])

## Perform K-fold CV to compare the best models
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=.4, random_state=42)

precision = dict()
recall = dict()

# Create average Precision-Recall Scores 
for key, value in classifier_dict.items():
    best_model = value["clf"].set_params(**value["best"])
    best_model.fit(X_train_train, y_train_train)
    scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    y_score = best_model.decision_function(X_train_test)
    classifier_dict[key]["pr_score"] = average_precision_score(y_train_test, y_score)
    precision[key], recall[key], _ = precision_recall_curve(y_train_test, y_score)
    print("Method ", key, "Scores ", scores)


colors = ['navy', 'turquoise', 'darkorange']

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []

for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

for key, color in zip(classifier_dict.keys(), colors):
    l, = plt.plot(recall[key], precision[key], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for {0} (area = {1:0.2f})'
                  ''.format(key, classifier_dict[key]["pr_score"]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for Multi Model Assessment')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()