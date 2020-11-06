import numpy as np
import warnings
import sklearn.metrics as metrics
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
    array_digits_largevis= readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/mnist_2D_LargeVis.txt', 70000, 2)
    array_fashion_largevis=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/fashion_mnist_2D_LargeVis.txt' ,70000 , 2)
    array_digits_trimap=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/mnist_2D_Trimap.txt' ,70000 , 2)
    array_fashion_trimap=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/fashion_mnist_2D_Trimap.txt' ,70000 , 2)
    array_digits_tsne=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/mnist_2D_TSNE.txt' ,70000 , 2)
    array_fashion_tsne=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/fashion_mnist_2D_TSNE.txt' ,70000 , 2)
    array_digits_umap=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/mnist_2D_UMAP.txt' ,70000 , 2)
    array_fashion_umap=readEmbeddings('/home/alexk101/Documents/Research_2020/tests/All_2D_Test_Output/with_first_line/fashion_mnist_2D_UMAP.txt' ,70000 , 2)

    embeddings.append([array_digits_largevis, array_fashion_largevis])
    embeddings.append([array_digits_trimap, array_fashion_trimap])
    embeddings.append([array_digits_tsne, array_fashion_tsne])
    embeddings.append([array_digits_umap, array_fashion_umap])

def calculateTestResults():
    for x in embeddings:
        silh=metrics.silhouette_score(x[0], digits_labels)
        davd=metrics.davies_bouldin_score(x[0], digits_labels)
        results.append([silh, davd])
        print("Processed a result")
        silh2=metrics.silhouette_score(x[1], fashion_labels)
        davd2=metrics.davies_bouldin_score(x[1], fashion_labels)
        results.append([silh2, davd2])
        print("Processed a result")

def printTestResults():
    for y in range(4):
        x=embeddings[y]
        x2=embeddings[y+1]

        if(y==0):
            print("LargeVis Results- \n")
        elif(y==1):
            print("Trimap Results- \n")
        elif(y==2):
            print("TSNE Results- \n")
        else:
            print("UMAP Results- \n")

        print("MNIST Digits Scores: silhouette:", x[0], "davies_bouldin:", x[1])
        print("Fashion MNIST Scores: silhouette:", x2[0], "davies_bouldin:", x2[1],"\n")
    
    print("Analysis complete!")


loadTestEmbeddings()
calculateTestResults()
printTestResults()
