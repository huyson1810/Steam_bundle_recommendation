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

FIn = snap.TFIn("graph/steam.graph")
G = snap.TUNGraph.Load(FIn)

def get_status(G):
	print('Number of Nodes: %d' % G.GetNodes()) #99288
	print('Number of Edges: %d' % G.GetEdges()) #5153209

	UserDegDict = {}
	GameDegDict = {}
	deg_sum = 0
	user_deg_sum = 0
	game_deg_sum = 0
	count_user = 0
	count_game = 0
	min_deg_user = 10000
	min_deg_game = 10000
	for node in G.Nodes():
		deg = node.GetDeg()
		deg_sum += deg
		
		if node.GetId() in user_node_array:	
			min_deg_user = min(min_deg_user, deg)
			if deg<1:
				count_user+=1
			user_deg_sum += deg
			if deg in UserDegDict.keys():
				UserDegDict[deg] += 1
			else:
				UserDegDict[deg] = 1
		else:
			min_deg_game = min(min_deg_game, deg)
			if deg<1:
				count_game+=1
			game_deg_sum += deg
			if deg in GameDegDict.keys():
				GameDegDict[deg] += 1
			else:
				GameDegDict[deg] = 1
	print("disconnected user nodes %d" % count_user)
	print("disconnected game nodes %d" % count_game)
	print("min degree of user %d" % min_deg_user)
	print("min degree of game %d " % min_deg_game)
	print("sum of degrees %d" % deg_sum)
	print("sum of degree of user nodes %d "% user_deg_sum)
	print("sum of degree of game nodes %d " % game_deg_sum)
	print("average", float(user_deg_sum/float(len(user_node_array))) )
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