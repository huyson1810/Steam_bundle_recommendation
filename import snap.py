import snap
import matplotlib.pyplot as plt 

FIn = snap.TFIn("graph/steam.graph")
G = snap.TUNGraph.Load(FIn)
print G.GetNodes(), G.GetEdges()
ls = []
for ni in G.Nodes():
	id = ni.GetId()
	if id < 600000 and ni.GetDeg() > 500:
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


def getDeg():
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

getDeg()
# snap.SaveEdgeList(G, "graph/steam500.edgelist")