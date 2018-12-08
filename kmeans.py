import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
from sklearn.manifold import TSNE
import time
from sklearn.cluster import KMeans
from ggplot import theme_bw
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

community_file = 'emd/louvain_user_nocutoff_weighted.txt'
emb_file = 'emd/steam_new_p1_q1.emd'

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
            #     break
    return np.array(game_embeddings), np.array(user_embeddings), np.array(game_ids), np.array(user_ids)


def read_community():
    with open(community_file, 'r') as f:
        f.readline()
        f.readline()
        count=0
        dct_node = {}
        dct_cluster = {}
        for row in f:
            row = row.translate(None, '[]')
            # cluster = [float(x) for x in row.split(',')]
            for x in row.split(','):
                dct_node[int(x)] = count
                if count in dct_cluster.keys():
                    dct_cluster[count].append(int(x)) 
                else:
                    dct_cluster[count]= [int(x)]
            count+=1
    return dct_node, dct_cluster
    
def get_emb_mtx(lst, X, ids):
    result = []
    for i in lst:
        idx = list(ids).index(i)
        result.append(X[idx,:])
    return result

def main():
    X_game, X_user, game_ids, user_ids = read_emd()
    dct_node, dct_cluster = read_community()
    # print(dct_cluster.keys())
    # x = [key for key in dct_cluster.keys() if len(dct_cluster[key])>800]
    for key in dct_cluster.keys():
        print(len(dct_cluster[key]))
    cluster_lst = []
    for key in dct_cluster.keys():
        nodes = dct_cluster[key]
        max_sil = 0
        opt_k = 0
        if len(nodes)>1000:
            print("cluster id %d"%key)
            X_cluster = get_emb_mtx(nodes, X_user, user_ids)
            scores = []
            ks = [5, 10, 15, 30, 50, 100, 150, 200, 250]
            for k in ks:

                kmeans = KMeans(n_clusters=k, random_state=0).fit(X_cluster)
                # inertia = kmeans.inertia_
                cluster_labels = kmeans.labels_
                sil = silhouette_score(X_cluster, cluster_labels)
                scores.append(sil)
                if sil>max_sil:
                    max_sil = sil
                    opt_k = k
                # sample_silhouette_values = silhouette_samples(X_cluster, cluster_labels)
                # fig, (ax1, ax2) = plt.subplots(1, 2)
                # fig.set_size_inches(18, 7)
                # y_lower = 100
                # print(key, k, sil)
                # for i in range(k):
                #     # Aggregate the silhouette scores for samples belonging to
                #     # cluster i, and sort them

                #     ith_cluster_silhouette_values = \
                #     sample_silhouette_values[cluster_labels == i]

                #     ith_cluster_silhouette_values.sort()

                #     size_cluster_i = ith_cluster_silhouette_values.shape[0]
                #     y_upper = y_lower + size_cluster_i

                #     color = cm.nipy_spectral(float(i) / float(k))
                #     ax1.fill_betweenx(np.arange(y_lower, y_upper),
                #                       0, ith_cluster_silhouette_values,
                #                       facecolor=color, edgecolor=color, alpha=0.7)

                #     # Label the silhouette plots with their cluster numbers at the middle
                #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                #     # print("text", k, i)
                #     # Compute the new y_lower for next plot
                #     y_lower = y_upper + 100  # 10 for the 0 samples
                # ax1.axvline(x=sil, color="red", linestyle="--")
                # ax1.set_xticks([-0.1, 0, 0.2, 0.4])
                # ax1.set_yticks([])
                # ax1.set_xlabel("The silhouette coefficient values")
                # ax1.set_ylabel("Cluster label")
                # # plt.axvline(x=silhouette_avg, color="red", linestyle="--")
                
                # # ax1.show()
                # colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
                # ax2.scatter(X_cluster[:][0], X_cluster[:][1], marker='.', s=30, lw=0, alpha=0.7,
                #             c=colors, 
                #             edgecolor='k')

                # # Labeling the clusters
                # centers = kmeans.cluster_centers_
                # # Draw white circles at cluster centers
                # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                #             c="white", alpha=1, s=200, edgecolor='k')

                # for i, c in enumerate(centers):
                #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                #                 s=50, edgecolor='k')

                # ax2.set_title("The visualization of the clustered data.")
                # ax2.set_xlabel("Feature space for the 1st feature")
                # ax2.set_ylabel("Feature space for the 2nd feature")

                # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                #               "with n_clusters = %d" % k),
                #              fontsize=14, fontweight='bold')
                # plt.show()
            cluster_lst.append((len(nodes), opt_k))
            print("optimal k", key, opt_k, max_sil)
            plt.plot(ks, scores, marker='s', label='cluster %d' %key)
            plt.plot(opt_k, max_sil, marker='o', markersize=3, color="red")
    plt.title("k clusters for kmeans")
    plt.xlabel('number of clusters')
    plt.ylabel('silhouette score')
    plt.legend()
    plt.show()
    cluster_lst.sort(key=lambda x: x[0])
    plt.plot([x[0] for x in cluster_lst], [x[1] for x in cluster_lst])
    plt.title("optimal k value vs number of nodes")
    plt.xlabel('number of nodes')
    plt.ylabel('optimal k value')
    plt.show()

        # print(k , sil_sum)
        # plt.legend()
        # plt.plot(x, inertias, label='k=%d, sum=%f'%(k,sil_sum))
            


    # for k in range(20):


if __name__ == '__main__':
    main()