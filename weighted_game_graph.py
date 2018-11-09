import json
from pprint import pprint
import ast
import snap
import sys
import time
import pickle


def get_weighted_game_graph_status():
	FIn = snap.TFIn("graph/steam_weight_game_1000.graph")
	G = snap.TNEANet.Load(FIn)
	attr='weight'
	print(G.GetNodes())
	print(G.GetEdges())
	count=0
	for edge in G.Edges():
		value = G.GetIntAttrDatE(edge, attr)
		if value>1:
			count+=1
	print(count)

if __name__ == '__main__':
	get_weighted_game_graph_status()