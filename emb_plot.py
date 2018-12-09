import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
from sklearn.cluster import KMeans
from ggplot import theme_bw



emb_file = 'emd/steam_new_p1_q1.emd'
# emb_file = 'emd/steam_new_p1_q0.5.emd'
# emb_file = 'emd/steam_new_p1_q2.emd'

# community_file = 'emd/louvain_game_nocutoff_weighted.txt'
# community_file = 'emd/louvain_user_nocutoff_weighted.txt'
community_file = 'emd/kmeans_louvain_user.txt'

def read_emd():
	with open(emb_file, 'r') as f:
		row0 = f.readline()
		# n = int(row0.split(' ')[0])
		# k = int(row0.split(' ')[1])
		user_embeddings = []
		game_embeddings = []
		game_ids = []
		user_ids = []
		count=0
		for row in f:
			nid = int(row.split(' ')[0])

			embed = [float(x) for x in row.split(' ')[1:]]
			if nid<600000:
				game_embeddings.append(embed)
				game_ids.append(nid)
			else:
				user_embeddings.append(embed)
				user_ids.append(nid)
			# count+=1
			# if count==10000:
			# 	break
	return np.array(game_embeddings), np.array(user_embeddings), np.array(game_ids), np.array(user_ids)

def read_community():
	with open(community_file, 'r') as f:
		f.readline()
		f.readline()
		count=0
		dct = {}
		for row in f:
			row = row.translate(None, '[]')
			for x in row.split(','):
				# print(x)
				dct[int(x)] = count
			count+=1
	# print(dct)
	return dct
		

def get_plot(X, ids, cluster_dct):

	new_X = []
	new_ids = []
	for (i,nid) in enumerate(ids):
		if nid in cluster_dct.keys():
			new_X.append(X[i, :])
			new_ids.append(nid)
	X = np.array(new_X)
	ids = np.array(new_ids)


	print(X.shape, ids.shape)
	feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

	df = pd.DataFrame(X,columns=feat_cols)

	n_sne = 480000

	rndperm = ids
	# print(len(rndperm), rndperm)

	# rndperm_o = np.random.permutation(df.shape[0])
	# time_start = time.time()




	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	# tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
	tsne_results = tsne.fit_transform(df.values)
	df_tsne = df.copy()
	df_tsne['x-tsne'] = tsne_results[:,0]
	df_tsne['y-tsne'] = tsne_results[:,1]
	# print(df_tsne)
	print(max(tsne_results[:,0]), min(tsne_results[:,0]), max(tsne_results[:,1]), min(tsne_results[:,1]))

	kmeans = KMeans(n_clusters=10, random_state=0).fit(tsne_results)

	# ids = ids[rndperm[:n_sne]]
	# print(ids[0], cluster_dct[ids[0]])
	labels = [cluster_dct[i] for i in ids]
	y = np.array(labels)
	print("y", y, y.shape)
	# y = np.array(kmeans.labels_)


	df_tsne['label'] = y
	df_tsne['label'] = df_tsne['label'].apply(lambda i: str(i))


	chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
	        + geom_point(size=5,alpha=0.5) \
			+ theme_bw()\
			+ ggtitle("tSNE dimensions of steam graph embeddings")

	# ggsave()
	print(chart)

def main():
	X_game, X_user, game_ids, user_ids = read_emd()
	cluster_dct = read_community()
	# print(cluster_dct)

	(n_game,k) = X_game.shape
	(n_user, k) = X_user.shape
	print(n_game, n_user, k)
	
	# get_plot(X_game, game_ids, cluster_dct)
	get_plot(X_user, user_ids, cluster_dct)

	



if __name__ == '__main__':
	main()
	# pass








# 