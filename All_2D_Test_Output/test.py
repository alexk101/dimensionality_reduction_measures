import numpy as np
import warnings
import trimap as tri
import sklearn.metrics as metrics
import sklearn.manifold as manifold
import time
import progressbar as pb
from sklearn.datasets import fetch_openml
warnings.filterwarnings("ignore")

# Written by: Alexander Kiefer
# Email: alkiefer@iu.edu
# Indiana University Bloomington
# Collaborators: Md Khaledur Rahman
# Email: morahma@iu.edu

mnistFashion=fetch_openml(name="Fashion-MNIST")
mnistDigits=fetch_openml("mnist_784", version=1)

fashion_labels=mnistFashion['target']
digits_labels=mnistDigits['target']
fashion_hd_data=mnistFashion['data']
digits_hd_data=mnistDigits['data']

embeddings=[]
results = []

def readEmbeddings(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    N = int(firstline.strip().split()[0])
    X = [[0]*dim for i in range(nodes)]
    currentLine=0
    for line in embfile.readlines():
        tokens = line.strip().split()
        nodeid = currentLine
        currentLine=currentLine+1
        x = []
        for j in range(1, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[nodeid] = x
    embfile.close()
    return np.array(X)


def loadTestEmbeddings():
    array_digits_largevis= readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/mnist_2D_LargeVis.txt', 70000, 2)
    array_fashion_largevis=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/fashion_mnist_2D_LargeVis.txt' ,70000 , 2)
    array_digits_trimap=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/mnist_2D_Trimap.txt' ,70000 , 2)
    array_fashion_trimap=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/fashion_mnist_2D_Trimap.txt' ,70000 , 2)
    array_digits_tsne=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/mnist_2D_TSNE.txt' ,70000 , 2)
    array_fashion_tsne=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/fashion_mnist_2D_TSNE.txt' ,70000 , 2)
    array_digits_umap=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/mnist_2D_UMAP.txt' ,70000 , 2)
    array_fashion_umap=readEmbeddings('/home/alexk101/dimensionality_reduction_measures/All_2D_Test_Output/with_first_line/fashion_mnist_2D_UMAP.txt' ,70000 , 2)

    embeddings.append([array_digits_largevis, array_fashion_largevis])
    embeddings.append([array_digits_trimap, array_fashion_trimap])
    embeddings.append([array_digits_tsne, array_fashion_tsne])
    embeddings.append([array_digits_umap, array_fashion_umap])

def calculateTestResults():
    for x in pb.progressbar(iter(embeddings), redirect_stdout=True):
        silh=metrics.silhouette_score(x[0], digits_labels)
        davd=metrics.davies_bouldin_score(x[0], digits_labels)
        globalStruct=tri.TRIMAP(verbose=False).global_score(digits_hd_data, x[0])
        localStruct=manifold.trustworthiness(digits_hd_data, x[0])
        results.append([silh, davd, globalStruct, localStruct])
        print("Processed a result")

        silh2=metrics.silhouette_score(x[1], fashion_labels)
        davd2=metrics.davies_bouldin_score(x[1], fashion_labels)
        globalStruct2=tri.TRIMAP(verbose=False).global_score(fashion_hd_data, x[1])
        localStruct2=manifold.trustworthiness(fashion_hd_data, x[1])
        results.append([silh2, davd2, globalStruct2, localStruct2])
        print("Processed a result")

def printTestResults():

    f = open("demofile3.txt", "w")

    for y in range(0,8,2):
        x=results[y]
        x2=results[y+1]

        if(y==0):
            m1="LargeVis Results- \n"
            print(m1)
            f.write(m1)
        elif(y==1):
            m2="Trimap Results- \n"
            print(m2)
            f.write(m2)
        elif(y==2):
            m3="TSNE Results- \n"
            print(m3)
            f.write(m3)
        else:
            m4="UMAP Results- \n"
            print(m4)
            f.write(m4)

        mc1="MNIST Digits Scores: silhouette: "+ str(x[0])+ "    |  davies bouldin: "+ str(x[1])+ "    | global structure: "+ str(x[2])+ "    |  local structure: "+ str(x[3])+"\n"
        print(mc1)
        f.write(mc1)
        mc2="Fashion MNIST Scores: silhouette: "+ str(x2[0]) +"  |  davies bouldin: "+ str(x2[1])+ "    | global structure: "+ str(x2[2])+ "    |  local structure: "+ str(x2[3])+"\n"
        print(mc2)
        f.write(mc2)
    
    print("Analysis complete!")
    f.close()

loadTestEmbeddings()
calculateTestResults()
printTestResults()
