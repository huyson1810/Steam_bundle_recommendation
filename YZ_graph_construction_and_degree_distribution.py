'''
(1) Construct reconstructed graph from original graph by removing certain nodes
(2) Degree distribution in original and reconstructed user-game graph
(3) Construct projected graph from reconstructed graph
(4) Degree distribution in projected user and game graph
'''
import snap
import matplotlib.pyplot as plt 

'''
Construct modified graph by removing:
(1) users that play more than 100 games
(2) games taht are played by more than 1000 users
(3) users who do not play any game and games that are not played by any user after (1) and (2)
'''
def construct():
	FIn = snap.TFIn("../graph/steam.graph")
	G = snap.TUNGraph.Load(FIn)
	print G.GetNodes(), G.GetEdges()

	ls = []
	for ni in G.Nodes():
		id = ni.GetId()
		if id >= 600000 and ni.GetDeg() > 100:
			ls.append(id)
		elif id < 600000 and ni.GetDeg() > 1000:
			ls.append(id)
	for i in ls:
		G.DelNode(i)
	print G.GetNodes(), G.GetEdges()

	ls = []
	for ni in G.Nodes():
		id = ni.GetId()
		if ni.GetDeg() == 0:
			ls.append(id)
	for i in ls:
		G.DelNode(i)

	print G.GetNodes(), G.GetEdges()

	FOut = snap.TFOut("../graph/steam_user100_game1000.graph")
	G.Save(FOut)
	FOut.Flush()


'''
Helper function
Increase counts in degree distribution dictionary dic[deg] by 1
'''
def update(d, deg):
	if deg in d:
		d[deg] += 1
	else:
		d[deg] = 1

'''
Helper function
Return dictionaries key : val - node degree : number of nodes with given degree for:
(1) both user and game nodes
(2) user nodes only
(3) game nodes only
'''
def getDeg(G):
	d = snap.TIntIntH()
	user = snap.TIntIntH()
	game = snap.TIntIntH()
	for ni in G.Nodes():
		deg = ni.GetDeg()
		update(d, deg)
		if ni.GetId() >= 600000:
			update(user, deg)
		else:
			update(game, deg)
	return d, user, game

'''
Helper function
Return two lists 
x = [node degree]
y = [number of nodes with given degree]
with y sorted in increasing order
'''
def getXY(d):
	d.SortByDat()
	x = []
	y = []
	it = d.BegI()
	for i in range(len(d)):
		x.append(it.GetKey())
		y.append(it.GetDat())
		it.Next()
	return x, y

'''
Helper function
Make plot
'''
def plot(d, user, game, title):
	X1, Y1 = getXY(d)
	X2, Y2 = getXY(user)
	X3, Y3 = getXY(game)
	plt.loglog(X1, Y1, color = 'y', label = 'User + Game Nodes')
	plt.loglog(X2, Y2, color = 'b', label = 'User Nodes')
	plt.loglog(X3, Y3, color = 'r', label = 'Game Nodes')
	plt.xlabel('Node Degree (log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title(title)
	plt.legend()
	plt.show()

'''
Plot degree distribution of all nodes, user nodes and game nodes
in the original and reconstructed steam user-game graphs
'''
def degreeDistribution():
	### read original and reonstructed graph
	FIn = snap.TFIn("../graph/steam.graph")
	Go = snap.TUNGraph.Load(FIn)
	FIn = snap.TFIn("../graph/steam_user100_game1000.graph")
	Gn = snap.TUNGraph.Load(FIn)
	print Go.GetNodes(), Go.GetEdges()
	print Gn.GetNodes(), Gn.GetEdges()

	d1, user1, game1 = getDeg(Go)
	d2, user2, game2 = getDeg(Gn)
	plot(d1, user1, game1, 'Original Graph')
	plot(d2, user2, game2, 'Reconstructed Graph')

'''
Helper function
Return a dictionary key : val - node degree : number of nodes with given degree
'''
def getDeg2(G):
	d = snap.TIntIntH()
	for ni in G.Nodes():
		deg = ni.GetDeg()
		update(d, deg)
	return d

'''
Helper function
Plot degree distribution in projected user graph and game graphs
'''
def plot2(d1, d2, title = 'Projected Graph'):
	X1, Y1 = getXY(d1)
	X2, Y2 = getXY(d2)
	plt.loglog(X1, Y1, color = 'b', label = 'User Nodes')
	plt.loglog(X2, Y2, color = 'r', label = 'Game Nodes')
	plt.xlabel('Node Degree (log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title(title)
	plt.legend()
	plt.show()

'''
Construct undirected unweighted graph from nodes and their adjacent lists 
represented as dic key : val - node id : (dic of neighbor node id : edge weight)
'''
def constructGraph(dic):
	G = snap.TUNGraph().New()
	for v, val in dic.iteritems():
		if not G.IsNode(v): G.AddNode(v)
		for w in val.keys():
			if not G.IsNode(w): G.AddNode(w)
			G.AddEdge(v, w)
	print snap.GetClustCf(G, -1)
	return G

'''
Construct projected graph from steam user-game graph, 
and plot degree distribution.
'''
def generate_steam_game_user_weight_graph():
	FIn = snap.TFIn("../graph/steam_user100_game1000.graph")
	G = snap.TUNGraph.Load(FIn)

	game_dct = {}
	user_dct = {}
	# add edges
	count=0
	for node in G.Nodes():
		NId = node.GetId()
		if NId < 600000:			
			ki = node.GetDeg()
			neid = []
			for i in range(ki):
				neid.append(node.GetNbrNId(i))
			neid = sorted(neid)
			for i in range(len(neid)):
				for j in range(i+1,len(neid)):
					if neid[i] in user_dct:
						temp_dic = user_dct[neid[i]]
						if neid[j] in temp_dic:
							temp_dic[neid[j]]+=1
						else:
							temp_dic[neid[j]]=1
					else:
						user_dct[neid[i]] = {}
						user_dct[neid[i]][neid[j]]=1
			# print(user_dct)	
		else:
			ki = node.GetDeg()
			neid = []
			for i in range(ki):
				neid.append(node.GetNbrNId(i))
			neid = sorted(neid)
			for i in range(len(neid)):
				for j in range(i+1,len(neid)):
					if neid[i] in game_dct:
						temp_dic = game_dct[neid[i]]
						if neid[j] in temp_dic:
							temp_dic[neid[j]]+=1
						else:
							temp_dic[neid[j]]=1
					else:
						game_dct[neid[i]] = {}
						game_dct[neid[i]][neid[j]]=1
	G = constructGraph(user_dct)
	d1 = getDeg2(G)
	print G.GetNodes(), G.GetEdges()
	G = constructGraph(game_dct)
	d2 = getDeg2(G)
	print d1.Len()
	print d2.Len()
	plot2(d1, d2)
	print G.GetNodes(), G.GetEdges()
	return game_dct, user_dct

if __name__ == '__main__':
	# construct()
	degreeDistribution()
	generate_steam_game_user_weight_graph()