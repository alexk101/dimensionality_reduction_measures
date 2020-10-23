from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from scipy.io import mmread,mminfo
from scipy.sparse import csr_matrix
import sys
import matplotlib.pyplot as plt
import networkx as nx
import random, math
import numpy as np
import warnings
from sklearn.datasets import fetch_openml
warnings.filterwarnings("ignore")

mnistFashion=fetch_openml(name="Fashion-MNIST")
mnistDigits=fetch_openml("mnist_784", version=1)

fashion_labels=mnistFashion.labels
digits_labels=mnistDigits.labels

def readEmbeddings(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    N = int(firstline.strip().split()[0])
    X = [[0]*dim for i in range(nodes)]
    for line in embfile.readlines():
        tokens = line.strip().split()
        nodeid = int(tokens[0])-1
        x = []
        for j in range(1, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[nodeid] = x
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

embeddings=[]

array_digits_largevis= readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/mnist_2D_LargeVis.txt', 70000, 2)
array_fashion_largevis=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/fashion_mnist_2D_LargeVis.txt' ,70000 , 2)
array_digits_trimap=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/mnist_2D_Trimap.txt' ,70000 , 2)
array_fashion_trimap=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/fashion_mnist_2D_Trimap.txt' ,70000 , 2)
array_digits_tsne=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/mnist_2D_TSNE.txt' ,70000 , 2)
array_fashion_tsne=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/fashion_mnist_2D_TSNE.txt' ,70000 , 2)
array_digits_umap=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/mnist_2D_UMAP.txt' ,70000 , 2)
array_fashion_umap=('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/fashion_mnist_2D_UMAP.txt' ,70000 , 2)


embeddings.append([array_digits_largevis, array_fashion_largevis])
embeddings.append([array_digits_trimap, array_fashion_trimap])
embeddings.append([array_digits_tsne, array_fashion_tsne])
embeddings.append([array_digits_umap, array_fashion_umap])

results = []

for x in embeddings:
    results.append([metrics.silhouette_score(x[0], gy), metrics.davies_bouldin_score(x[1], gy)])

print("silhouette:", shil, "davies_bouldin:", davd)

print("Visualization complete!")
