'''
Map the user node id from community graph to the id that are used in bundle recommendation.
'''
import ast
import pickle
import csv

def constructMap():
	original_user_ids = []
	with open("../data/australian_users_items.json") as f:
		for line in f:
			data = ast.literal_eval(line)
			user_id = data['user_id']
			original_user_ids.append(user_id)
	with open('graph_id_to_user_name', 'w') as f:
	    for item in original_user_ids:
			f.write("%s\n" % item)
	f.close()

	dic = {}
	user_id_lookup = pickle.load(open('../data/processed_data/user_id_lookup','rb'))
	for user_key, user_name in user_id_lookup.items():
		dic[user_name] = user_key
	with open('user_name_to_user_key', 'wb') as filedump:
	    pickle.dump(dic, filedump, protocol=pickle.HIGHEST_PROTOCOL)
	filedump.close()
	return original_user_ids, dic

def getCommunities():
	communities = []
	with open("../data/louvain_user_nocutoff_weighted") as f:
		for line in f:
			community = []
			vec = line.split(" ")
			for i in vec:
				community.append(int(i))
			communities.append(community)
	print len(communities)
	return communities

def getCommunityUsers(communities, original_user_ids, dic):
	C = []
	for community in communities:
		l = []
		for user in community:
			user_name = original_user_ids[user - 600000]
			if user_name in dic:
				user_key = dic[user_name]
				l.append(user_key)
		C.append(l)
	with open('user_key_communities.csv', 'w') as f:
		writer = csv.writer(f, delimiter=',')
		for l in C:
			writer.writerow(l)

if __name__ == '__main__':
	original_user_ids, dic = constructMap()
	communities = getCommunities()
	getCommunityUsers(communities, original_user_ids, dic)