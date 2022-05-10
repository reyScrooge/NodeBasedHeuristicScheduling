import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


with open('stats_new.pickle', 'rb') as handle:
    stats = pickle.load(handle)

nodeIDs = [i for i in stats.keys()][1:]

heuristics = ['actconsdiving', 'coefdiving', 'conflictdiving', 'crossover', 'distributiondivin', 'farkasdiving', 'fracdiving',
              'guideddiving', 'linesearchdiving', 'localbranching', 'pscostdiving', 'rens', 'rins', 'mutation', 'dins', 'trustregion',
              'veclendiving']

# first line of matrix is nothing
matrix = np.zeros((1000, 17), dtype=float)

# this is how the grade is compute
for i in range(1, 1001):
    heus = stats[i]['heuStats'].keys()
    for heu in heus:
        heuIndex = heuristics.index(heu)
        execTime =  stats[i]['heuStats'][heu][4]
        matrix[i-1][heuIndex] = 1/execTime
heu_matrix = preprocessing.normalize(matrix, axis = 1, norm='l1')

# normalize features
feature_matrix = np.zeros((1000, 36), dtype=float)
for i in range(1, 1001):
    feature_matrix[i-1] = stats[i]['features']
mu = np.average(feature_matrix, axis=0)
std = np.std(feature_matrix, axis=0)
for j, e in enumerate(std):
    if e < 1e-3: std[j] = 1
feature_matrix = (feature_matrix-mu)/std

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, heu_matrix, test_size=0.1)

labels = []
# n clusters
number_of_clusters = 200
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(X_train)
# labels matrix: kmeans.labels_
indices = []
for i in range(number_of_clusters):
    indices.append([j for j in range(len(y_train)) if kmeans.labels_[j] == i])

average_grade = []
for index in indices:
    average_grade.append(y_train[index].sum(axis=0)/len(index))
# average_grade.append(y_train[indices_0].sum(axis = 0)/len(indices_0))
# average_grade.append(y_train[indices_1].sum(axis = 0)/len(indices_1))
# average_grade.append(y_train[indices_2].sum(axis = 0)/len(indices_2))

prediction = kmeans.predict(X_test)

for i in range(0,100):
    print("Test Node: ", i)
    temp = average_grade[prediction[i]]
    # print("prediction: ",average_grade)
    # print("grade:", y_test[i])
    # print("order of prediction: ", np.argsort(temp).tolist().reverse())
    # print("order of grade: ", np.argsort(y_test[i]).tolist().reverse())
    print(len(list(set(np.argsort(temp)[10:]).intersection(np.argsort(y_test[i])[10:]))) / 7)



