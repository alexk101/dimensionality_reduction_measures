import trimap
import numpy
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1)
output = open(r"Trimap_2D.txt","w")

embedding = trimap.TRIMAP().fit_transform(mnist.data)

output.write("70000 2 \n")
numpy.savetxt("Trimap_2D.txt",embedding)
