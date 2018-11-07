import snap
import matplotlib.pyplot as plt



user_node_array = []
with open('graph/user_node.txt', 'r') as f:
    for line in f:
        user_node_array.append(int(line))

game_node_array = []
with open('graph/game_node.txt', 'r') as f:
    for line in f:
        game_node_array.append(int(line)) 

FIn = snap.TFIn("steam.graph")
G = snap.TUNGraph.Load(FIn)

def get_status(G):
	print('Number of Nodes: %d' % G.GetNodes()) #99288
	print('Number of Edges: %d' % G.GetEdges()) #5153209

	UserDegDict = {}
	GameDegDict = {}
	for node in G.Nodes():
		deg = node.GetDeg()
		if node.GetId() in user_node_array:	
			if deg in UserDegDict.keys():
				UserDegDict[deg] += 1
			else:
				UserDegDict[deg] = 1
		else:
			if deg in GameDegDict.keys():
				GameDegDict[deg] += 1
			else:
				GameDegDict[deg] = 1
	Userlists = sorted(UserDegDict.items())
	x_User, y_User = zip(*Userlists)
	plt.loglog(x_User,y_User, color = 'y', label = 'User')
	Gamelists = sorted(GameDegDict.items())
	x_Game, y_Game = zip(*Gamelists)
	plt.loglog(x_Game,y_Game, linestyle = 'dashed', color = 'r', label = 'Game')
	plt.xlabel('Node Degree (Log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title('Degree Distribution of User and Game')
	plt.legend()
	plt.show()

get_status(G)