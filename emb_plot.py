import numpy as np
# from sklearn import datasets
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ggplot import *
from sklearn.manifold import TSNE
# import hyperspy as hs
# import h5py
import time
from sklearn.cluster import KMeans



filename = '/Users/yulian/Desktop/Courses/CS224W/project/node2vec-master/emb/steam500p1q1.emd'
# f = h5py.File(filename, 'r')
# print(f)
def read_emd():
	with open(filename, 'r') as f:
		row0 = f.readline()
		# n = int(row0.split(' ')[0])
		# k = int(row0.split(' ')[1])
		embeddings = []
		count=0
		for row in f:
			nid = int(row.split(' ')[0])
			embed = [float(x) for x in row.split(' ')[1:]]
			embeddings.append(embed)
			count+=1
			if count==10000:
				break
	return np.array(embeddings)

def main():
	X = read_emd()
	(n,k) = X.shape
	print(n,k)
	
	feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

	df = pd.DataFrame(X,columns=feat_cols)
	rndperm = np.random.permutation(df.shape[0])


	n_sne = 5000

	rndperm = np.random.permutation(df.shape[0])
	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
	df_tsne = df.loc[rndperm[:n_sne],:].copy()
	df_tsne['x-tsne'] = tsne_results[:,0]
	df_tsne['y-tsne'] = tsne_results[:,1]
	# print(df_tsne)
	print(max(tsne_results[:,0]), min(tsne_results[:,0]), max(tsne_results[:,1]), min(tsne_results[:,1]))

	kmeans = KMeans(n_clusters=10, random_state=0).fit(tsne_results)
	y = np.array(kmeans.labels_)
	df_tsne['label'] = y
	df_tsne['label'] = df_tsne['label'].apply(lambda i: str(i))


	chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
	        + geom_point(size=20,alpha=0.8) \
	        + ggtitle("tSNE dimensions of steam graph embeddings")
	print(chart)

	



if __name__ == '__main__':
	main()
	# pass








