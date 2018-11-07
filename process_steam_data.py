import json
from pprint import pprint
import ast
import snap
import sys

def generate_steam_graph():
	G = snap.TUNGraph.New()
	user_node_array = []
	user_node_id = 600000
	game_node_array = [] #10978
	# min_game_id = sys.maxint #10
	# max_game_id = 0 #530720
	count1=0
	count2=0
	with open("data/australian_users_items.json") as f:
		for line in f:
			data = ast.literal_eval(line)
			user_id = data['user_id']
			item_count = int(data['items_count'])
			# if item_count>900:
			# 	count1+=1
			# if item_count>1000:
			# 	count2+=1
			G.AddNode(user_node_id)
			items = data['items']
			for item in items:
				item_id = int(item['item_id'])
				min_game_id = min(item_id, min_game_id)
				max_game_id = max(item_id, max_game_id)
				game_node_array.append(item_id)
				if not G.IsNode(item_id):
					G.AddNode(item_id)
					game_node_array.append(item_id)
				G.AddEdge(user_node_id, item_id)
			user_node_array.append(user_node_id)
			user_node_id+=1
	print(len(game_node_array))
	print(min_game_id, max_game_id)
	with open('graph/user_node.txt', 'w') as f:
	    for item in user_node_array:
	        f.write("%d\n" % item)

	with open('graph/game_node.txt', 'w') as f:
	    for item in game_node_array:
	        f.write("%d\n" % item)

	FOut = snap.TFOut("steam.graph")
	G.Save(FOut)
	FOut.Flush()

def generate_steam_user_graph():
	FIn = snap.TFIn("graph/steam.graph")
	G = snap.TUNGraph.Load(FIn)

	user_node_array = []
	with open('graph/user_node.txt', 'r') as f:
	    for line in f:
	        user_node_array.append(int(line))

	game_node_array = []
	with open('graph/game_node.txt', 'r') as f:
	    for line in f:
	        game_node_array.append(int(line)) 

	# TODO
	



if __name__ == '__main__':
	generate_steam_graph()
	# generate_steam_user_graph()