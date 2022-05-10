import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


with open('stats_new.pickle', 'rb') as handle:
    stats = pickle.load(handle)

# nodeIDs = [i for i in stats.keys()][1:]


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


labels = []
# n clusters
number_of_clusters = 200
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(heu_matrix)
# labels matrix: kmeans.labels_
indices = []
nodeIDs = [i for i in range(1000)]

X_train, X_test, y_train, y_test, nodeIDs_train, nodeIDs_test = train_test_split(feature_matrix, kmeans.labels_, nodeIDs,test_size=0.1)
knn = KNeighborsClassifier(n_neighbors = number_of_clusters).fit(X_train, y_train)

# collect indices for each cluster
for i in range(number_of_clusters):
    indices.append([j for j in range(len(y_train)) if kmeans.labels_[j] == i])

average_grade = {}
for index in range(len(indices)):
    if len(indices[index]) == 0:
        continue

    # get original ids of nodes of each cluster
    temp_indices = []
    for i in indices[index]:
        temp_indices.append(nodeIDs_train[i])

    average_grade[index] = heu_matrix[temp_indices].sum(axis=0)/len(indices[index])

prediction = knn.predict(X_test)

for i in range(0,100):
    print("Test Node: ", i)
    temp = average_grade[prediction[i]]
    orogin_index = nodeIDs_test[i]
    # print("prediction: ",average_grade)
    # print("grade:", y_test[i])
    # print("order of prediction: ", np.argsort(temp))
    # print("order of grade: ", np.argsort(heu_matrix[orogin_index]))
    print(len(list(set(np.argsort(temp)[10:]).intersection(np.argsort(heu_matrix[orogin_index])[10:])))/7)



