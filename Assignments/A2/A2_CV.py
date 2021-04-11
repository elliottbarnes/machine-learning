from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve, precision_recall_curve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# read in the file name specified
file_name = sys.argv[1]

# read in the data from the file in the current directory
# not currently configured to read in from command line
df = pd.read_csv(file_name, sep="\t")
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# set parameters for KNN that we want to optimize
grid_params = {
    'n_neighbors': [5,11,19, 29],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# setup grid cv with f1 scoring
gs = GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    scoring = 'f1'
)

# do the grid cv on the x and y set
gs_r = gs.fit(x,y)

# print best score, best params, and results of the grid cv
print("This is the best f1 score: ", gs_r.best_score_)
print("This is the best parameters for KNN: ",gs_r.best_params_)
print("The grid score results are shown below...")
print(pd.DataFrame(gs_r.cv_results_))
print("End of grid score results... displaying graphs now")

# set the KNN classifier to the best params
classifier = gs_r.best_estimator_

cv = StratifiedKFold()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# create and plot ROC curve
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(x, y)):
    classifier.fit(x.iloc[train], y.iloc[train])
    viz = plot_roc_curve(classifier, x.iloc[test], y.iloc[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="COMP 3202 Assignment 2 - Part 5")
ax.legend(loc="lower right")
plt.show()

# create and plot PR Curve
prs = []
aucs = []
mean_recall = np.linspace(0, 1, 100)

plt.figure(figsize=(18 , 13))
i = 0
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x.iloc[train], y.iloc[train]).predict_proba(x.iloc[test])
    # Compute PR curve and area the curve
    precision, recall, thresholds = precision_recall_curve(y.iloc[test], probas_[:, 1])
    prs.append(np.interp(mean_recall, precision, recall))
    pr_auc = auc(recall, precision)
    aucs.append(pr_auc)
    plt.plot(recall, precision, lw=3, alpha=0.5, label='PR Fold %d (AUCPR = %0.2f)' % (i+1, pr_auc))
    i += 1

plt.plot([0, 1], [1, 0], linestyle='--', lw=3, color='k', label='Luck', alpha=.8)
mean_precision = np.mean(prs, axis=0)
mean_auc = auc(mean_recall, mean_precision)
std_auc = np.std(aucs)
plt.plot(mean_precision, mean_recall, color='navy',
         label=r'Mean (AUCPR = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=4)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall' ,  fontweight = "bold" , fontsize=30)
plt.ylabel('Precision',fontweight = "bold" , fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend( prop={'size':15} , loc = 0)
plt.show()