totalModGain = 0

class Community:

	def __init__(self, node):
		self.Id = node.getcId()
		self.nodes = set()
		self.totalWeight = 0
		self.inWeight = 0
		self.addNode(node)

	def getInWeight(self):
		return self.inWeight

	def getId(self):
		return self.Id

	def getNodes(self):
		return self.nodes

	def addNode(self, node):
		node.setcId(self.Id)
		self.nodes.add(node.getId())
		self.totalWeight += node.getTotalWeight()
		self.inWeight += node.getSelfWegiht() + self.weightWithNode(node)

	def removeNode(self, node):
		self.nodes.remove(node.getId())
		self.totalWeight -= node.getTotalWeight()
		self.inWeight -= node.getSelfWegiht() + self.weightWithNode(node)

	def weightWithNode(self, node):
		kiin = 0
		for nbr, weight in node.getNeighbors():
			if nbr.getId() in self.nodes:
				kiin += 2 * weight
		return kiin

	def addNeighborWeight(self, weight):
		self.totalWeight += weight

	def modularityWNode(self, node, m):
		kiin = self.weightWithNode(node)
		ki = node.getTotalWeight()
		return kiin / m / 2 - ki * self.totalWeight / m / m / 2

	def modularityWONode(self, node, m):
		kiin = self.weightWithNode(node)
		ki = node.getTotalWeight()
		return -kiin / m / 2 + ki * (self.totalWeight - ki) / m / m / 2


class Node:

	def __init__(self, ID, weight):
		self.dic = {}
		self.neighbors = []
		self.nId = ID
		self.cId = ID
		self.sWeight = weight
		self.oWeight = 0

	def addNeighbor(self, node, weight, community):
		Id = node.getcId()
		i = len(self.neighbors)
		if Id in self.dic:
			i = self.dic[Id]
			self.neighbors[i][1] += weight
		else:
			self.dic[Id] = i
			self.neighbors.append([node, weight])
		self.oWeight += weight
		community.addNeighborWeight(weight)

	def getNeighbors(self):
		return self.neighbors

	def getSelfWegiht(self):
		return self.sWeight

	def getOutWegiht(self):
		return self.oWeight

	def getTotalWeight(self):
		return self.sWeight + self.oWeight

	def setcId(self, Id):
		self.cId = Id

	def getcId(self):
		return self.cId

	def getId(self):
		return self.nId


class Graph:

	def __init__(self, path):
		self.IdNodeMap = []
		self.nodes = []
		self.communities = []
		self.m = 0.0
		file = open(path, 'r');
		V = int(file.readline())
		for i in range(V):
			node = Node(i, 0)
			self.IdNodeMap.append([i])
			self.nodes.append(node)
			self.communities.append(Community(node))

		for line in file:
			v, w, l = (int(s) for s in line.split(','))
			self.m += l
			self.nodes[v].addNeighbor(self.nodes[w], l, self.communities[v])
			self.nodes[w].addNeighbor(self.nodes[v], l, self.communities[w])

	def getNode(self, Id):
		return self.nodes[Id]

	def louvain(self):
		change = True
		while change:
			change = self.modularityOpt()
			if change:
				while self.modularityOpt(): continue
				self.communityAgg()
			
	def modularityOpt(self):
		global totalModGain
		change = False
		for node in self.nodes:
			cId = node.getcId()
			maxcId = -1
			maxMGain = 0
			dQDi = self.communities[cId].modularityWONode(node, self.m)
			seen = set()
			for nbr in node.getNeighbors():
				ncId = nbr[0].getcId()
				if cId == ncId or ncId in seen: continue
				seen.add(ncId)
				dQiC = self.communities[ncId].modularityWNode(node, self.m)
				if dQDi + dQiC > maxMGain:
					maxMGain = dQDi + dQiC
					maxcId = ncId

			if maxMGain > 0:
				totalModGain += maxMGain
				change = True
				self.communities[maxcId].addNode(node)
				self.communities[cId].removeNode(node)

		return change

	def communityAgg(self):
		newNodes = []
		newCommunities = []
		newIdNodeMap = []
		dic = {}
		i = 0
		for community in self.communities:
			if len(community.getNodes()) == 0: continue
			nodes = community.getNodes()
			ls = []
			for nodeId in nodes:
				ls.extend(self.IdNodeMap[nodeId])
			newIdNodeMap.append(ls)
			node = Node(i, community.getInWeight())
			newNodes.append(node)
			newCommunities.append(Community(node))
			dic[community.getId()] = i
			i += 1

		for community in self.communities:
			if len(community.getNodes()) == 0: continue
			Id = community.getId()
			newId = dic[Id]
			for nodeId in community.getNodes():
				for nbr, w in self.getNode(nodeId).getNeighbors():
					nId = nbr.getcId()
					if nId == Id: continue
					newnId = dic[nId]
					newNodes[newId].addNeighbor(newNodes[newnId], w, newCommunities[newnId])

		self.nodes = newNodes
		self.communities = newCommunities
		self.IdNodeMap = newIdNodeMap
		print "Communities after one iteration:"
		print newIdNodeMap
		print "Self-edge weight and degree of supernode:"
		for i in range(len(self.nodes)):
			print self.nodes[i].getSelfWegiht(), self.nodes[i].getOutWegiht()

g = Graph("graph2.csv")

g.louvain()
print "Total modularity gain is", totalModGain