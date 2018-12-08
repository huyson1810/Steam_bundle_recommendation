'''
Plot the information during bundle generation.
'''

import matplotlib.pyplot as plt

colors = [(0, 0, 0), (0.3, 0.1, 0.7), (0.5, 0.1, 0.5), (0.7, 0.1, 0.3), 
          (0.9, 0.1, 0.1), (0.7, 0.3, 0.1), (0.5, 0.5, 0.1), (0.3, 0.7, 0.1), 
          (0.1, 0.9, 0.1), (0.1, 0.7, 0.3), (0.1, 0.5, 0.5), (0.1, 0.3, 0.7)]

'''
Extract information (bundle size, rank and aggregate diversity) from command line output.
'''
def getPar(path, size):
	total_bundle_size = []
	total_pre_score = []
	avg_bundle_size = []
	agg_diversity = []

	with open(path) as file:
		for line_number, line in enumerate(file):
			ls = line.split(',')
			if line_number % 5 == 1:
				total_bundle_size.append([int(n) for n in ls])
			elif line_number % 5 == 2:
				total_pre_score.append([float(n) for n in ls])
			elif line_number % 5 == 3:
				avg_bundle_size.append([float(n) for n in ls])
				bundle_size = total_bundle_size[len(total_bundle_size) - 1]
				agg_diversity.append([bundle_size[i] / (float(ls[i]) * size) for i in range(len(ls))])

	return total_bundle_size, total_pre_score, avg_bundle_size, agg_diversity

'''
Plot aggregated bundle size.
'''
def plot_total_bundle_size(total_bundle_size, sizes, size):
	bundle_size = total_bundle_size[0]
	
	plt.plot(sizes[0 : len(bundle_size)], bundle_size, 'o', color = colors[0])
	plt.plot(sizes[0 : len(bundle_size)], bundle_size, color = colors[0], label= 'All users')
	for i in range(1, len(total_bundle_size)):
		bundle_size = total_bundle_size[i]
		plt.plot(sizes[0 : len(bundle_size)], bundle_size, 'o', color = colors[i] )
		plt.plot(sizes[0 : len(bundle_size)], bundle_size, linestyle = 'dashed', color = colors[i], label= 'community ' + str(i))
	plt.ylabel('Aggregated bundle size from ' + size + ' groups of users')
	plt.xlabel('Number of users per group')
	plt.legend(loc='lower left', ncol=3)
	plt.show()

'''
Plot bundle rank.
'''
def plot_pref_score(total_pre_score, sizes, size):
	pre_score = total_pre_score[0]
	
	plt.plot(sizes[0 : len(pre_score)], pre_score, 'o', color = colors[0])
	plt.plot(sizes[0 : len(pre_score)], pre_score, color = colors[0], label= 'random')
	for i in range(1, len(total_pre_score)):
		pre_score = total_pre_score[i]
		plt.plot(sizes[0 : len(pre_score)], pre_score, 'o', color = colors[i] )
		plt.plot(sizes[0 : len(pre_score)], pre_score, linestyle = 'dashed', color = colors[i], label= 'community ' + str(i))
	plt.ylabel('Average rank from ' + size + ' groups of users')
	plt.xlabel('Number of users per group')
	# plt.legend(loc='upper left', ncol=3)
	plt.show()

'''
Plot average bundle size.
'''
def plot_avg_bundle_size(avg_bundle_size, sizes, size):
	bundle_size = avg_bundle_size[0]
	
	plt.plot(sizes[0 : len(bundle_size)], bundle_size, 'o', color = colors[0])
	plt.plot(sizes[0 : len(bundle_size)], bundle_size, color = colors[0], label= 'random')
	for i in range(1, len(avg_bundle_size)):
		bundle_size = avg_bundle_size[i]
		plt.plot(sizes[0 : len(bundle_size)], bundle_size, 'o', color = colors[i] )
		plt.plot(sizes[0 : len(bundle_size)], bundle_size, linestyle = 'dashed', color = colors[i], label= 'community ' + str(i))
	plt.ylabel('Average bundle size from ' + size + ' groups of users')
	plt.xlabel('Number of users per group')
	# plt.legend(loc='lower left', ncol=3)
	plt.show()

'''
Plot aggregate diversity.
'''
def plot_agg_diversity(agg_diversity, sizes, size):
	agg_div = agg_diversity[0]
	# 
	
	plt.plot(sizes[0 : len(agg_div)], agg_div, 'o', color = colors[0])
	plt.plot(sizes[0 : len(agg_div)], agg_div, color = colors[0], label= 'random')
	for i in range(1, len(agg_diversity)):
		bundle_size = agg_diversity[i]
		plt.plot(sizes[0 : len(agg_div)], agg_div, 'o', color = colors[i] )
		plt.plot(sizes[0 : len(agg_div)], agg_div, linestyle = 'dashed', color = colors[i], label= 'community ' + str(i))
	plt.ylabel('Aggregate diversity from ' + size + ' groups of users')
	plt.xlabel('Number of users per group')
	# plt.legend(loc='lower left', ncol=3)
	plt.show()

def plot1():
	sizes = [1, 3, 10, 30, 100]
	size = 10
	total_bundle_size, total_pre_score, avg_bundle_size, agg_diversity = getPar("bundle_score", size)
	plt.xlim(-10, 110)
	plt.ylim(0, 40)
	plot_total_bundle_size(total_bundle_size, sizes, str(size))
	plt.xlim(-10, 110)
	plt.ylim(4, 15)
	plot_pref_score(total_pre_score, sizes, str(size))
	plt.xlim(-10, 110)
	plt.ylim(0, 5)
	plot_avg_bundle_size(avg_bundle_size, sizes, str(size))

def plot2():
	sizes = [1, 10]
	size = 80
	total_bundle_size, total_pre_score, avg_bundle_size, agg_diversity = getPar("bundle_score_2", size)
	plt.xlim(0, 11)
	plt.ylim(0, 200)
	plot_total_bundle_size(total_bundle_size, sizes, str(size))
	plt.xlim(0, 11)
	plt.ylim(4, 10)
	plot_pref_score(total_pre_score, sizes, str(size))
	plt.xlim(0, 11)
	plt.ylim(0, 5)
	plot_avg_bundle_size(avg_bundle_size, sizes, str(size))
	# plt.xlim(0, 11)
	# plt.ylim(0, 0.8)
	# plot_agg_diversity(agg_diversity, sizes, str(size))

plot1()
plot2()