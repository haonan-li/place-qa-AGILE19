import json
import numpy as np
from sklearn import cluster
from sklearn import metrics
from Levenshtein import *
from tqdm import tqdm
import math
import sys,time


def load_code(fcode):
    f = open(fcode,'r')
    C = []
    S = set()
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        C.append((line[0],line[1]))
        S.add(line[1])
    n_observe = len(C)
    code = list(S)
    n_code = len(S)
    print (n_code)
    X = np.zeros((n_observe,n_code))

    print ('compute distance matrix')
    for i in tqdm(range(n_observe)):
        for j in range(n_code):
            X[i][j] = jaro(C[i][1],code[j])
    print ('done')
    f.close()

    return C,X


def kmeans_cluster(X,k=10):
    estimator = cluster.KMeans(n_clusters=k, n_jobs=3)
    estimator.fit(X)
    label_pred = estimator.labels_
    score = metrics.calinski_harabaz_score(X,label_pred)
    return label_pred, score


def output(fout,C,label,k=100,score=0):
    result = [ [] for i in range(k) ]
    for i in range(len(C)):
        result[label[i]].append(C[i])
    with open (fout,'w') as f:
        json.dump(result,f,indent=4)
        f.write('\n'+'score: '+str(score)+'\n')


def main():

    for sent_c in ['concat_code', 'query_code', 'answer_code']:
        fcode = '../data/' + sent_c + '.txt'
        C,X = load_code(fcode)

        for k in range(2,10):
            label,score = kmeans_cluster(X,k)
            fout = 'result/'+sent_c+'/kmeans' + str(k)
            print (sent_c,"n_clusters=", k,"score:",score)
            output(fout,C,label,k,score)

main()
