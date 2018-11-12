import json
from pprint import pprint
import ast
import snap
import sys
import time
import pickle
import csv
import matplotlib.pyplot as plt


def get_weighted_game_graph_status():
	start = time.time()
	FIn = snap.TFIn("graph/steam_weight_user_limit_100.graph")
	G = snap.TNEANet.Load(FIn)

	# G = snap.GetMxWcc(G)

	ClustCf = snap.GetClustCf(G, 1000)
	print("clustering coefficient: %f" % ClustCf)	

	print("numer of nodes %d" % G.GetNodes())
	print("number of edges %d" % G.GetEdges())
	
	NWcc = snap.GetMxWcc(G).GetNodes()
	NScc = snap.GetMxScc(G).GetNodes()
	print("numer of nodes in max wcc %d" % NWcc)
	print("number of nodes in max scc %d" % NScc)


	deg_dict = {}
	sum_deg = 0
	count = 0
	min_deg = 10000
	for node in G.Nodes():
		deg = node.GetDeg()
		min_deg = min(min_deg, deg)
		if deg<1:
			count+=1
		sum_deg += deg
		if deg in deg_dict.keys():
			deg_dict[deg]+=1
		else:
			deg_dict[deg]=1
	print("disconnected nodes %d" % count)
	print("minimum node degree %d "% min_deg)

	print("sum of degree: %d " % sum_deg)

	avg_deg = sum_deg/float(G.GetNodes())
	print("average degree: %f" % avg_deg)

	lst = sorted(deg_dict.items())
	x, y = zip(*lst)
	plt.loglog(x, y, linestyle = 'dashed', color = 'r', label = 'Game')
	plt.xlabel('Node Degree (Log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title('Degree Distribution of Game')
	plt.legend()
	plt.show()

	weight_dict = {}
	sum_weight = 0
	attr = 'weight'
	for edge in G.Edges():
		weight = G.GetIntAttrDatE(edge, attr)
		sum_weight += weight
		if weight in weight_dict.keys():
			weight_dict[weight]+=1
		else:
			weight_dict[weight]=1

	print("sum of weight: %d" % sum_weight)

	avg_weight = sum_weight/float(G.GetEdges())
	print("average weight: %d" % avg_weight)



	lst = sorted(weight_dict.items())
	x, y = zip(*lst)
	plt.loglog(x, y, linestyle = 'dashed', color = 'r', label = 'Game')
	plt.xlabel('Edge weight (Log)')
	plt.ylabel('Proportion of Edges with a Given Weight (log)')
	plt.title('Edge weight Distribution of Game')
	plt.legend()
	plt.show()



def write_graph_to_csv(name):
	start = time.time()
	FIn = snap.TFIn("graph/steam_weight_user_limit_100.graph")
	G = snap.TNEANet.Load(FIn)

	print("finished loading: ", time.time()-start)

	attr='weight'

	# put node ids into consecutive integers
	node_dict = {}
	nid_count = 0
	for node in G.Nodes():
		nid = node.GetId()
		if nid not in node_dict.keys():
			node_dict[nid] = nid_count
			nid_count += 1

	print("finished generating node id dict: ", time.time()-start)

	with open('csv/weighted_%s_limit_100.csv'%name, mode='w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow([str(len(node_dict.keys()))])
		# count = 0
		for edge in G.Edges():
			srcnid = node_dict[edge.GetSrcNId()]
			dstnid = node_dict[edge.GetDstNId()]
			# print(edge.GetSrcNId(), edge.GetDstNId(), srcnid, dstnid)
			value = G.GetIntAttrDatE(edge, attr)
			row = [str(srcnid), str(dstnid), str(value)]
			writer.writerow(row)
			# count+=1

	print("done: ", time.time()-start)


def generate_steam_edge_list():
	FIn = snap.TFIn("graph/steam.graph")
	G = snap.TUNGraph.Load(FIn)

	G = snap.GetMxWcc(G)

	user_node_array = [] #88310
	with open('graph/user_node.txt', 'r') as f:
	    for line in f:
	        user_node_array.append(int(line))

	game_node_array = [] #10978
	with open('graph/game_node.txt', 'r') as f:
	    for line in f:
	        game_node_array.append(int(line)) 

	with open('graph/steam_edge_list.csv', 'w') as f:
		writer = csv.writer(f, delimiter=',')
		for edge in G.Edges():
			# eid = edge.GetId()
			id1 = edge.GetSrcNId()
			id2 = edge.GetDstNId()
			if id1 in user_node_array:
				row = [str(id1), 'g'+str(id2)]
			else:
				row = [str(id2), 'g'+str(id1)]
			writer.writerow(row)


if __name__ == '__main__':
	write_graph_to_csv('user')
	# get_weighted_game_graph_status()