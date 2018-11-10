import json
from pprint import pprint
import ast
import snap
import sys
import time
import pickle
import csv


def get_weighted_game_graph_status():
	start = time.time()
	FIn = snap.TFIn("graph/steam_weight_game_10000.graph")
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

	with open('weighted_game.csv', mode='w') as f:
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

if __name__ == '__main__':
	get_weighted_game_graph_status()