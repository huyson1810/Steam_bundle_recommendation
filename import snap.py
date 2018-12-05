import snap
import matplotlib.pyplot as plt 


def construct():
	FIn = snap.TFIn("../steam.graph")
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
	return G


def getDeg(G):
	d = snap.TIntIntH()
	n = 0
	m = 0
	for ni in G.Nodes():
		deg = ni.GetDeg()
		if ni.GetId() >= 600000:
			n += deg * deg
		else:
			m += deg * deg
		if deg in d:
			d[deg] += 1
		else:
			d[deg] = 1

	print n, m

def plot(d):
	d.SortByDat()
	x = []
	y = []
	it = d.BegI()
	for i in range(len(d)):
		x.append(it.GetKey())
		y.append(it.GetDat())
		it.Next()
	plt.loglog(x, y, color = 'b', label = 'degree distribution')
	plt.show()

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

def getDeg2(G):
	d = snap.TIntIntH()
	for ni in G.Nodes():
		deg = ni.GetDeg()
		if deg in d:
			d[deg] += 1
		else:
			d[deg] = 1
	return d

def constructGraph(dic):
	G = snap.TUNGraph().New()
	for v, val in dic.iteritems():
		if not G.IsNode(v): G.AddNode(v)
		for w in val.keys():
			if not G.IsNode(w): G.AddNode(w)
			G.AddEdge(v, w)
	return G

''' generate steam game user weight graph new'''
def generate_steam_game_user_weight_graph():
	FIn = snap.TFIn("../graph/steam_user100_game1000.graph")
	G = snap.TUNGraph.Load(FIn)

	# G_game = snap.PUNGraph.New()
	# G_user = snap.PUNGraph.New()
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

generate_steam_game_user_weight_graph()