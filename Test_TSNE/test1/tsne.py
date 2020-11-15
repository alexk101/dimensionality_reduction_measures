'''
Created on Mar 18, 2020

@author: alexk101
'''
from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml("mnist_784", version=1)
output = open(r"TSNE_2D.txt","w")
standardized_data = StandardScaler().fit_transform(mnist.data)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(standardized_data)

output.write("70000 2 \n")
np.savetxt("TSNE_2D.txt",tsne_results)
output.close()
print("Test Complete")
