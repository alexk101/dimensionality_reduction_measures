"""
UMAP on the MNIST Digits dataset
--------------------------------

A simple example demonstrating how to use UMAP on a larger
dataset such as MNIST. We first pull the MNIST dataset and
then use UMAP to reduce it to only 2-dimensions for
easy visualisation.

Note that UMAP manages to both group the individual digit
classes, but also to retain the overall global structure
among the different digit classes -- keeping 1 far from
0, and grouping triplets of 3,5,8 and 4,7,9 which can
blend into one another in some cases.
"""
import umap
from sklearn.datasets import fetch_openml
import numpy

mnist = fetch_openml("mnist_784", version=1)
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist['data'])

output = open(r"/home/alexk101/Documents/Research_2020/tests/Test_Umap/Test1/UMAP_2D.txt","w")
output.write("70000 2 \n")
numpy.savetxt("/home/alexk101/Documents/Research_2020/tests/Test_Umap/Test1/UMAP_2D.txt",embedding)
output.close()
print("Test Complete")
