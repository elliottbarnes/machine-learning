import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import random

def predict(df_indexes, class_df):
    d = [0, 0]
    for i in df_indexes:
        d[class_df.iloc[i]] += 1
    return d.index(max(d))

## initialize constants and filename
FILE_NAME = sys.argv[1]
K_VALUE = int(sys.argv[2])
U_VALUE = int(sys.argv[3])

## set the threshold for U based on the number of lines in the file
threshold = sum(1 for line in open(FILE_NAME)) - 1

## handle exceptions & invalid args
if (U_VALUE > threshold):
    print("Number of unknown instances should be less than the number of observations in the file")
    sys.exit()

if (K_VALUE < 1 or K_VALUE > threshold-U_VALUE):
    print("Number of neighbours should be in the range [1, {}]".format(threshold-U_VALUE))
    sys.exit()

# create pandas data frame from input data
df = pd.read_csv(FILE_NAME, delimiter='\t', header=0)

# Separate the cols and rows to x and y
x = df.drop('Class', axis=1) 
y = df.Class
new_x = x.copy()
header_list = x.columns.values

# create unknown instance dataframe
ui = pd.DataFrame(columns=header_list)

# Get the random Unknown Instances
indexes = []
for i in range(0, U_VALUE):
    rand = random.randint(0, threshold-1)
    while (rand in indexes):
        rand = random.randint(0, threshold-1)
    indexes.append(rand)

    x_test = x.iloc[rand]   # random point to test and find class
    ui = ui.append(x_test)       # append point to uninstance data
    new_x = new_x.drop(rand)        # update x data without this point in it

# Perform KNN Algorithm to predict the class
correct = []
for idx in range(len(ui)):
    instance = ui.iloc[idx]
    distances = []
    for i in range(len(new_x)):
        current = new_x.iloc[i]
        distance = 0
        for j in range(1, len(instance)):
            if instance[j] != current[j]:
                distance += 1
        distances.append(distance)
    df_dists = pd.DataFrame(data=distances, index=new_x.index, columns=['dist'])
    df_nn = df_dists.sort_values(by=['dist'], axis=0)[:K_VALUE]
    nn_indexes = df_nn.index.values
    prediction = predict(nn_indexes, y)
    correct.append(df.loc[df['Sequence.id'] == instance['Sequence.id']]['Class'].values[0] == prediction)

correctness = correct.count(True) / U_VALUE * 100
print(f"K = {K_VALUE}\tPercent of Correctly Classified Instances = {correctness}%")
