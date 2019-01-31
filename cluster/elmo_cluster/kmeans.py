import numpy as np
import json
from sklearn import cluster
from sklearn import decomposition
from sklearn import metrics
import h5py
import re
import sys,time


def load_sent(fraw):
    f = open(fraw,'r')
    # M is the map from sentence to id
    M = dict()
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        if line[1] in M:
            M[line[1]].append(line[0])
        else:
            M[line[1]] = [line[0]]
    return M


def load_data(frep, M):

    f = h5py.File(frep,'r')
    X = []
    S = []
    for key in f.keys():
        try:
            rep = f[key].value
            rep = np.mean(rep,axis=0)
            for i in range(len(M[key])):
                X.append(rep)
                S.append([key,M[key][i]])
        except:
            continue
    X = np.array(X)
    print (X.shape)

    return S, X


def kmeans_cluster(data,k=10):
    estimator = cluster.KMeans(n_clusters=k)
    estimator.fit(data)
    label_pred = estimator.labels_
    score = metrics.calinski_harabaz_score(data, label_pred)
    return label_pred, score


def output(fout,S,label,k,score):
    result = [ [] for i in range(k) ]
    for i in range(len(S)):
        result[label[i]].append(S[i])
    with open (fout,'w') as f:
        json.dump(result,f,indent=4)
        f.write('\n'+'score: '+str(score)+'\n')


def main():

    for sent_c in ['query_noname','answer_noname','query','answer']:
        frep = '../hdf5/'+sent_c+'.hdf5'
        fraw = '../data/' + sent_c + '.txt'
        M = load_sent(fraw)
        S,X = load_data(frep,M)

        for k in range(3,25):
            label,score = kmeans_cluster(X,k)
            fout = 'result/'+sent_c+'/kmeans' + str(k)
            print (sent_c,"n_clusters=", k,"score:",score)
            output(fout,S,label,k,score)

main()
