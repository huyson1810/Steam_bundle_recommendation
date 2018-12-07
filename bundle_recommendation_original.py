import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


items_set=pickle.load(open('../data/processed_data/item_set','rb'))
bundle_item_map=pickle.load(open('../data/processed_data/bundle_item_map','rb'))
user_bundle_map=pickle.load(open('../data/processed_data/user_bundle_map','rb'))
user_item_map=pickle.load(open('../data/processed_data/user_item_map','rb'))
bundle_diversity_map=pickle.load(open('../data/processed_data/bundle_diversity_map','rb'))

item_data=pickle.load(open('../data/processed_data/all_items','rb'))
item_id_lookup = pickle.load(open('../data/processed_data/item_id_lookup','rb'))

item_name_map=pickle.load(open('../data/processed_data/item_name_map','rb'))

item_data_map=dict()
tags_set=set()
for item in item_data:
    item_data_map[int(item['appid'])]=item
    for tag in item['tags']:
        tags_set.add(tag)
tags_map=dict()
for i,tag in enumerate(tags_set):
    tags_map[tag]=i
def get_feat(tags):
    feat=np.zeros(len(tags_map))
    for tag in tags:
        feat[tags_map[tag]]=1
    return feat

all_data=[]

for user,bundles in user_bundle_map.items():
    for bundle in bundles:
        all_data.append((user,bundle))
        
all_item_data=[]
for user,items in user_item_map.items():
    for item in items:
        all_item_data.append((user,item))


import random
random.shuffle(all_data)
data_size=len(all_data)

# Training data for bundle for bpr model
training_data=all_data[:int(0.8*data_size)]
test_data=all_data[int(0.8*data_size):]

# Training data for items for bpr_item model
training_data_2=all_item_data[:int(0.8*len(all_item_data))]
test_data_2=all_item_data[int(0.8*len(all_item_data)):]


def check_tuple(tuple_1, tuple_2, user_bundle_map):
    return tuple_1[1] not in user_bundle_map[tuple_2[0]] and tuple_2[1] not in user_bundle_map[tuple_1[0]]

def graph_sampling(n_samples, training_data, user_bundle_map):
    sgd_users=[]
    sgd_pos_items, sgd_neg_items = [], []
    i=0
    while n_samples>0:
        if i%100000==0:
            print i
        i+=1
        tuple_1=training_data[np.random.randint(len(training_data))]
        tuple_2=training_data[np.random.randint(len(training_data))]
        iteration=100
        while not check_tuple(tuple_1, tuple_2, user_bundle_map):
            tuple_2=training_data[np.random.randint(len(training_data))]
            iteration-=1
            if iteration == 0:
                break
        if iteration==0:
            continue   
        sgd_neg_items.append(tuple_2[1])
        sgd_pos_items.append(tuple_1[1])
        sgd_users.append(tuple_1[0])
        
        sgd_neg_items.append(tuple_1[1])
        sgd_pos_items.append(tuple_2[1])
        sgd_users.append(tuple_2[0])
        n_samples-=2
    return sgd_users, sgd_pos_items, sgd_neg_items

# Generting training data for items through graph sampling.
sgd_train_users_items, sgd_train_pos_items, sgd_train_neg_items = graph_sampling(len(training_data_2)*30, training_data_2, user_item_map)


def get_test_data_items(test_data, train_data):
    users=[]
    pos_items=[]
    neg_items=[]
    train_dict, train_users, train_items  = data_to_dict(train_data)
    test_dict, test_users, test_items = data_to_dict(test_data)
    z = 0
    for i,user in enumerate(test_dict.keys()):
        if(i%1000==0):
            print i

        if user in train_users: 
            for pos_item in test_dict[user]:
                if pos_item in train_items:
                    for neg_item in train_items:
                        if neg_item not in test_dict[user] and neg_item not in train_dict[user]:
                            users.append(user)
                            pos_items.append(pos_item)
                            neg_items.append(neg_item)

    return users, pos_items, neg_items


def data_to_dict(data):
    data_dict = defaultdict(list)
    items = set()
    for (user, item) in data:
        data_dict[user].append(item)
        items.add(item)
    return data_dict, set(data_dict.keys()), items


test_users_cold, test_pos_items_cold, test_neg_items_cold = get_test_data_items(test_data_2, training_data_2)

import os
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,lib.cnmem=0.7,floatX=float32'


# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import theano, numpy
import theano.tensor as T
import time
import sys
from collections import defaultdict

class BPR_Item(object):

    def __init__(self, rank, n_users, n_items, lambda_u = 0.0025, lambda_i = 0.0025, lambda_j = 0.00025, lambda_d = 0.0025, lambda_p = 0.00025, lambda_bias = 0.0, learning_rate = 0.05):
        
        self._rank = rank
        self._n_users = n_users
        self._n_items = n_items
        self._lambda_u = lambda_u
        self._lambda_i = lambda_i
        self._lambda_j = lambda_j
        self._lambda_d = lambda_d
        self._lambda_p = lambda_p
        self._lambda_bias = lambda_bias
        self._learning_rate = learning_rate
        self._configure_theano()
        self._generate_train_model_function()
        self._generate_test_model_function()

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats. 
        """
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _generate_train_model_function(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        h = numpy.random.random((self._n_items, self._rank))
        b = numpy.random.random(self._n_items)
        
        self.W = theano.shared(numpy.random.random((self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(h.astype('float32'), name='H')
        self.B = theano.shared(b.astype('float32'), name='B')
        
        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + self.B[i]
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + self.B[j]
        x_uij = T.nnet.sigmoid(x_ui-x_uj)
        
        obj = T.sum(T.log(x_uij) - self._lambda_u * (self.W[u] ** 2).sum(axis=1) - 
                    self._lambda_i * (self.H[i] ** 2).sum(axis=1) - self._lambda_j * 
                    (self.H[j] ** 2).sum(axis=1) - self._lambda_bias * 
                    (self.B[i] ** 2 + self.B[j] ** 2) )
       
    
        cost = - obj

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [(self.W, self.W - self._learning_rate * g_cost_W), (self.H, self.H - self._learning_rate * g_cost_H), 
                   (self.B, self.B - self._learning_rate * g_cost_B) ]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)


    def train(self, s_users=None, s_pos_items=None, s_neg_items=None, batch_size=1000):
        """
          Trains the BPR Matrix Factorisation model using Stochastic
          Gradient Descent and minibatches over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.

          We first create a set of random samples from `train_data` for 
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples by
          batches of length `batch_size`, and run one iteration of gradient
          descent for the batch.
        """
        if len(s_pos_items) < batch_size:
            sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(s_pos_items)
            
        sgd_users, sgd_pos_items, sgd_neg_items = s_users, s_pos_items, s_neg_items
        n_sgd_samples = len(s_users)
        
        z = 0
        t2 = t1 = t0 = time.time()
        while (z+1)*batch_size < n_sgd_samples:
            self.train_model(
                sgd_users[z*batch_size: (z+1)*batch_size],
                sgd_pos_items[z*batch_size: (z+1)*batch_size],
                sgd_neg_items[z*batch_size: (z+1)*batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
        if n_sgd_samples > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
            sys.stderr.flush()
            
    def _generate_test_model_function(self):
        """
          Computes item predictions for `user_index`.
          Returns an array of prediction values for each item
          in the dataset.
        """
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + self.B[i]
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + self.B[j]
        x_uij = x_ui-x_uj
        
        self.test_model = theano.function(inputs=[u, i, j], outputs=x_uij)

    def test_bundle(self, sgd_users, sgd_pos_items, sgd_neg_items, batch_size=1000):
        """
          Computes the Area Under Curve (AUC) on `test_data`.

          `test_data` is an array of (user_index, item_index) tuples.

          During this computation we ignore users and items
          that didn't appear in the training data, to allow
          for non-overlapping training and testing sets.
        """
        
        auc_values = []
        z = 0
        t2 = t1 = t0 = time.time()
        n_sgd_samples = len(sgd_users)
        while (z+1)*batch_size < n_sgd_samples:
            pref_list=self.test_model(
                sgd_users[z*batch_size: (z+1)*batch_size],
                sgd_pos_items[z*batch_size: (z+1)*batch_size],
                sgd_neg_items[z*batch_size: (z+1)*batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            t1 = t2
            
            auc = np.sum([1.0 if a>0.0 else 0.0 for a in pref_list])
            auc /= batch_size
            
            auc_values.append(auc)
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z*batch_size), numpy.mean(auc_values)))
            sys.stderr.flush()
        
        sys.stderr.write("\n")
        sys.stderr.flush()
        return numpy.mean(auc_values)

bpr_item = BPR_Item(10, len(user_item_map.keys()), len(items_set))

#1 
bpr_item.train(s_users=sgd_train_users_items, s_pos_items=sgd_train_pos_items, s_neg_items=sgd_train_neg_items)

#2 
bpr_item.test_bundle(test_users_cold, test_pos_items_cold, test_neg_items_cold)

# Bundle model begins

# Generting training data for bundles through graph sampling.
sgd_users, sgd_pos_bundles, sgd_neg_bundles = graph_sampling(len(training_data)*30, training_data, user_bundle_map)

# Determining max bundle size to create bins for N
max_bundle_size=0
for bundle,items in bundle_item_map.items():
    if(len(items)>max_bundle_size):
        max_bundle_size=len(items)
print max_bundle_size
print len(items_set)

def get_items(bundle_id, max_bundle_size, index):
    item=list(bundle_item_map[bundle_id]);
    for i in range(len(item),max_bundle_size):
        item.append(index)
    return item

sgd_pos_items=[get_items(b_id, max_bundle_size, len(items_set)) for b_id in sgd_pos_bundles]
sgd_neg_items=[get_items(b_id, max_bundle_size, len(items_set)) for b_id in sgd_neg_bundles]

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
def compute_diversity_tags(app_data):
    l=len(app_data)
    app_data=[item_id_lookup[d] for d in app_data]
    count=0.0
    similarity=0.0
    for i in range(l):
        if app_data[i] in item_data_map:
            for j in range(i+1,l):
                if app_data[j] in item_data_map:
                    count+=1
                    similarity+=jaccard_similarity_score(get_feat(item_data_map[app_data[i]]['tags']),
                                                         get_feat(item_data_map[app_data[j]]['tags']))
    if count>0:
        return 1-(similarity/count)
    else:
        return 0.0

def compute_diversity_latent(app_data, H):
    l=len(app_data)
    count=0.0
    similarity=0.0
    for i in range(l):
            for j in range(i+1,l):
                    count+=1
                    similarity+=cosine_similarity(H[i],H[j])[0,0]
    if count>0:
        return 1-(similarity/count)
    else:
        return 0.0
    
def compute_diversity(app_data, H):
    if H is not None:
        return compute_diversity_latent(app_data, H)
    else :
        return compute_diversity_tags(app_data)

Gamma=bpr_item.H.eval()
bundle_diversity_map=dict()
for bundle,items in bundle_item_map.items():
    bundle_diversity_map[bundle]=compute_diversity_latent(list(items), bpr_item.H.eval())

#bundle_diversity_map=pickle.load(open('../../data/pickle/training_data/game_aus/bpr/bundle_diversity_map','rb'))

sgd_pos_len=[len(bundle_item_map[b_id]) for b_id in sgd_pos_bundles]
sgd_neg_len=[len(bundle_item_map[b_id]) for b_id in sgd_neg_bundles]
sgd_pos_diversity=[bundle_diversity_map[b_id] for b_id in sgd_pos_bundles]
sgd_neg_diversity=[bundle_diversity_map[b_id] for b_id in sgd_neg_bundles]

def get_test_data_bundles(test_data, train_data, n_items):
    users=[]
    pos_items=[]
    neg_items=[]
    n1=[]
    n2=[]
    pos_diversity=[]
    neg_diversity=[]
    train_dict, train_users, train_items  = data_to_dict(train_data)
    test_dict, test_users, test_items = data_to_dict(test_data)
    auc_values = []
    z = 0
    for i,user in enumerate(test_dict.keys()):
        if(i%1000==0):
            print i

        if user in train_users: 
            for pos_item in test_dict[user]:
                if pos_item in train_items:
                    for neg_item in train_items:
                        if neg_item not in test_dict[user] and neg_item not in train_dict[user]:
                            pos_diversity.append(bundle_diversity_map[pos_item])
                            neg_diversity.append(bundle_diversity_map[neg_item])
                            users.append(user)
                            pos_items.append(pos_item)
                            neg_items.append(neg_item)
                            n1.append(len(bundle_item_map[pos_item]))
                            n2.append(len(bundle_item_map[neg_item]))

    pos_items=[get_items(b_id, max_bundle_size, n_items) for b_id in pos_items]
    neg_items=[get_items(b_id, max_bundle_size, n_items) for b_id in neg_items]
    return users, pos_items, neg_items, n1, n2, pos_diversity, neg_diversity

test_users, test_pos_items, test_neg_items, test_n1, test_n2, test_pos_diversity, test_neg_diversity= get_test_data_bundles(test_data, training_data, len(items_set))

print np.shape(bpr_item.H.eval())
H_item=bpr_item.H.eval()
H_item = np.concatenate((H_item,np.zeros((1,np.shape(H_item)[1]))),axis=0)
H_item=np.array(H_item).astype('float32')
print np.shape(H_item)

print np.shape(bpr_item.B.eval())
B_item=bpr_item.B.eval()
B_item = np.append(B_item,0)
B_item=np.array(B_item).astype('float32')
print np.shape(B_item)

import theano, numpy
import theano.tensor as T
import time
import sys
from collections import defaultdict

class BPR_Cold(object):

    def __init__(self, rank, bundle_size, n_users, n_items, lambda_u = 0.0025, lambda_i = 0.0025, lambda_j = 0.00025, lambda_d = 0.0025, lambda_p = 0.00025, lambda_bias = 0.0, learning_rate = 0.05):
        
        self._rank = rank
        self._bundle_rank = bundle_size + 1
        self._n_users = n_users
        self._n_items = n_items
        self._lambda_u = lambda_u
        self._lambda_i = lambda_i
        self._lambda_j = lambda_j
        self._lambda_d = lambda_d
        self._lambda_p = lambda_p
        self._lambda_bias = lambda_bias
        self._learning_rate = learning_rate
        self._configure_theano()
        self._generate_train_model_item_function()
        self._generate_test_model_function()

    def _configure_theano(self):
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'
    
    def _generate_train_model_item_function(self):
        u = T.lvector('u')
        i = T.lmatrix('i')
        j = T.lmatrix('j')
        n1 = T.lvector('n1')
        n2 = T.lvector('n2')
        di = T.dvector('di')
        dj = T.dvector('dj')
        
        self.W1 = bpr_item.W
        self.H1 = theano.shared(H_item.astype('float32'), name='H')
        self.B1 = theano.shared(B_item.astype('float32'), name='B')
        
        
        self.M1 = theano.shared(numpy.random.random((self._rank, self._rank)).astype('float64'), name='M1')
        self.M2 = theano.shared(numpy.random.random((self._rank, self._rank)).astype('float64'), name='M2')
        self.K = theano.shared(numpy.random.rand(), name='K')
        self.D = theano.shared(numpy.random.rand(), name='D')
        self.N = theano.shared(numpy.random.random(self._bundle_rank).astype('float32'), name='N')
        
        x_ui = T.dot(T.dot(self.W1[u],self.M2), T.dot(self.M1, self.H1[i].sum(axis=1).T/n1)).diagonal() + self.K*(self.B1[i].T/n1).T.sum(axis=1) + self.N[n1] + self.D*di
        x_uj = T.dot(T.dot(self.W1[u],self.M2), T.dot(self.M1, self.H1[j].sum(axis=1).T/n2)).diagonal() + self.K*(self.B1[j].T/n2).T.sum(axis=1) + self.N[n2] + self.D*dj

        x_uij = T.nnet.sigmoid(x_ui-x_uj)
        obj = T.sum(T.log(x_uij) - self._lambda_u * (self.M1 ** 2).sum() - \
                    self._lambda_u * (self.M2 ** 2).sum()  - self._lambda_d * (self.K**2) - self._lambda_d * (self.D**2)\
                    -self._lambda_p * (self.N[n2]**2) - self._lambda_p * (self.N[n1]**2))
        cost = - obj

        g_cost_M1 = T.grad(cost=cost, wrt=self.M1)
        g_cost_M2 = T.grad(cost=cost, wrt=self.M2)
        g_cost_K = T.grad(cost=cost, wrt=self.K)
        g_cost_N = T.grad(cost=cost, wrt=self.N)
        g_cost_D = T.grad(cost=cost, wrt=self.D)
        
        updates = [(self.M1, self.M1 - self._learning_rate * .001* g_cost_M1), (self.M2, self.M2 - self._learning_rate *.001* g_cost_M2), 
                   (self.K, self.K - self._learning_rate * .001*g_cost_K), (self.N, self.N - self._learning_rate *g_cost_N),
                  (self.D, self.D - self._learning_rate * g_cost_D)]

        self.train_model_item = theano.function(inputs=[u, i, j, n1, n2, di, dj], outputs=cost, updates=updates)

    def train(self, s_users=None, s_pos_items=None, s_neg_items=None, s_pos_len=None, s_neg_len=None,
             s_pos_diversity=None, s_neg_diversity=None,batch_size=1000):
        
        if len(s_users) < batch_size:
            sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(s_users)
        
        sgd_users, sgd_pos_items, sgd_neg_items = s_users, s_pos_items, s_neg_items
        n_sgd_samples = len(s_users)

        z = 0
        t2 = t1 = t0 = time.time()
        while (z+1)*batch_size < n_sgd_samples:
            
            self.train_model_item(
                sgd_users[z*batch_size: (z+1)*batch_size],
                sgd_pos_items[z*batch_size: (z+1)*batch_size],
                sgd_neg_items[z*batch_size: (z+1)*batch_size],
                s_pos_len[z*batch_size: (z+1)*batch_size],
                s_neg_len[z*batch_size: (z+1)*batch_size],
                s_pos_diversity[z*batch_size: (z+1)*batch_size],
                s_neg_diversity[z*batch_size: (z+1)*batch_size],
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
        if n_sgd_samples > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
            sys.stderr.flush()

    def _generate_test_model_function(self):
        u = T.lvector('u')
        i = T.lmatrix('i')
        j = T.lmatrix('j')
        n1 = T.lvector('n1')
        n2 = T.lvector('n2')
        di = T.dvector('di')
        dj = T.dvector('dj')
        
        x_ui = T.dot(T.dot(self.W1[u],self.M2), T.dot(self.M1, self.H1[i].sum(axis=1).T/n1)).diagonal() + self.K*(self.B1[i].T/n1).T.sum(axis=1) + self.N[n1] + self.D*di
        x_uj = T.dot(T.dot(self.W1[u],self.M2), T.dot(self.M1, self.H1[j].sum(axis=1).T/n2)).diagonal() + self.K*(self.B1[j].T/n2).T.sum(axis=1) + self.N[n2] + self.D*dj 
        
        x_uij = x_ui-x_uj
        self.test_model = theano.function(inputs=[u, i, j, n1, n2, di, dj], outputs=x_uij)


    def test_bundle(self, sgd_users, sgd_pos_items, sgd_neg_items, s_pos_len, s_neg_len, s_pos_diversity, s_neg_diversity, batch_size=1000):
        
        auc_values = []
        z = 0
        t2 = t1 = t0 = time.time()
        n_sgd_samples = len(sgd_users)
        while (z+1)*batch_size < n_sgd_samples:
            pref_list=self.test_model(
                sgd_users[z*batch_size: (z+1)*batch_size],
                sgd_pos_items[z*batch_size: (z+1)*batch_size],
                sgd_neg_items[z*batch_size: (z+1)*batch_size],
                s_pos_len[z*batch_size: (z+1)*batch_size],
                s_neg_len[z*batch_size: (z+1)*batch_size],
                s_pos_diversity[z*batch_size: (z+1)*batch_size],
                s_neg_diversity[z*batch_size: (z+1)*batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            t1 = t2
            
            auc = np.sum([1.0 if a>0.0 else 0.0 for a in pref_list])
            auc /= batch_size
            
            auc_values.append(auc)
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z*batch_size), numpy.mean(auc_values)))
            sys.stderr.flush()
        
        sys.stderr.write("\n")
        sys.stderr.flush()
        return numpy.mean(auc_values)

bpr_cold = BPR_Cold(10, max_bundle_size, len(user_bundle_map.keys()), len(items_set))

#3 
bpr_cold.train(s_users=sgd_users, s_pos_items=sgd_pos_items, s_neg_items=sgd_neg_items, 
          s_pos_len=sgd_pos_len, s_neg_len=sgd_neg_len, s_pos_diversity=sgd_pos_diversity, s_neg_diversity=sgd_neg_diversity)
#bpr_cold.train(s_users=sgd_users, s_pos_items=sgd_pos_items, s_neg_items=sgd_neg_items, 
#          s_pos_len=sgd_pos_len, s_neg_len=sgd_neg_len, s_pos_diversity=sgd_pos_diversity, s_neg_diversity=sgd_neg_diversity)

#4 
bpr_cold.test_bundle(test_users, test_pos_items, test_neg_items, test_n1, test_n2, test_pos_diversity, test_neg_diversity)

#Bundle Generation Through Greedy Model Begins

def generate_bundle(items_set, user, initial_size = 3, max_iteration = 1000, sample_size = 5):
    current_bundle = np.random.choice(list(items_set), initial_size)
    
    T=1000.0
    
    iteration = 0
    while iteration < max_iteration:
        iteration+=1
        curr_diversity = compute_diversity(current_bundle, Gamma)
        user_set=[]
        pos_item_set=[]
        actual_item_set=[]
        neg_item_set=[]
        pos_item_count=[]
        neg_item_count=[]
        pos_diversity=[]
        neg_diversity=[]
        
        
        candidate_items = set(np.random.choice(list(items_set), sample_size))
        
        for item in current_bundle:
            if item in candidate_items:
                candidate_items.remove(item)
        
    
        #Generating new bundles by adding and removing new items  
        for cand_item in candidate_items:
            #Add an item case
            if len(current_bundle)<10:
                user_set.append(user)    
                neg_item_set.append(add_bogus_items(current_bundle , max_bundle_size, len(items_set)))
                neg_item_count.append(len(current_bundle))
                neg_diversity.append(curr_diversity)         
                new_bundle=list(current_bundle)
                new_bundle.append(cand_item)
                pos_item_count.append(len(new_bundle))
                pos_diversity.append(compute_diversity(new_bundle, Gamma))
                actual_item_set.append(new_bundle)
                pos_item_set.append(add_bogus_items(new_bundle , max_bundle_size, len(items_set)))

            # Replace an item case
            for curr_item in current_bundle:
                user_set.append(user)
                
                neg_item_set.append(add_bogus_items(current_bundle , max_bundle_size, len(items_set)))
                neg_item_count.append(len(current_bundle))
                neg_diversity.append(curr_diversity)
                
                new_bundle=list(current_bundle)
                new_bundle.append(cand_item)
                new_bundle.remove(curr_item)
                pos_item_set.append(add_bogus_items(new_bundle , max_bundle_size, len(items_set)))
                actual_item_set.append(new_bundle)
                pos_item_count.append(len(new_bundle))
                pos_diversity.append(compute_diversity(new_bundle, Gamma))

        # Remove an item case
        if len(current_bundle)>2:
            for curr_item in current_bundle:
                user_set.append(user)

                neg_item_set.append(add_bogus_items(current_bundle , max_bundle_size, len(items_set)))
                neg_item_count.append(len(current_bundle))
                neg_diversity.append(curr_diversity)

                new_bundle=list(current_bundle)
                new_bundle.remove(curr_item)
                actual_item_set.append(new_bundle)
                pos_item_set.append(add_bogus_items(new_bundle , max_bundle_size, len(items_set)))
                pos_item_count.append(len(new_bundle))
                pos_diversity.append(compute_diversity(new_bundle, Gamma))
        
                
        pref_score = bpr_cold.test_model(user_set, pos_item_set, neg_item_set, pos_item_count, 
                                    neg_item_count, pos_diversity, neg_diversity)
                                    
        #print pref_score, pos_item_count, neg_item_count
        index = np.argmax(pref_score)
        #print "Pref Score ", pref_score[index]
        if(pref_score[index]>0):
            current_bundle = actual_item_set[index]
        else:
            prob = np.exp(pref_score[index]/T)
            if prob < .00001:
                break
            if np.random.rand() < prob:
                current_bundle = actual_item_set[index]
        T=T*0.9
    #print iteration
    return current_bundle

def add_bogus_items(bundle, max_bundle_size, index):
    item=list(bundle);
    for i in range(len(item),max_bundle_size):
        item.append(index)
    return item

def remove_bogus_items(bundle, max_bundle_size, index):
    item=list(bundle);
    i=0
    while i< len(bundle):
        if bundle[i]==index:
            break
        i+=1
    return bundle[:i]

def get_bundle_rank(user, new_bundle, bundle_item_map, bundle_diversity_map):
    user_set=[]
    pos_item_set=[]
    neg_item_set=[]
    pos_item_count=[]
    neg_item_count=[]
    pos_diversity=[]
    neg_diversity=[]
    
    bundle_diversity=compute_diversity(new_bundle, Gamma)
    for bundle_id,bundle in bundle_item_map.items():
        user_set.append(user)
        pos_item_set.append(add_bogus_items(bundle, max_bundle_size, len(items_set)))
        neg_item_set.append(add_bogus_items(new_bundle, max_bundle_size, len(items_set)))
        pos_item_count.append(len(bundle))
        neg_item_count.append(len(new_bundle))
        pos_diversity.append(bundle_diversity_map[bundle_id])
        neg_diversity.append(bundle_diversity)
        
    pref_score = bpr_cold.test_model(user_set, pos_item_set, neg_item_set, pos_item_count, 
                                    neg_item_count, pos_diversity, neg_diversity)
  
    rank = np.sum([1.0 if p>0 else 0.0 for p in pref_score])
    return rank

sizes=[10]
diversities=[]
scores=[]
bundle_sizes=[]
for size in sizes:
    aggregate_diversity=set()
    pred_score=[]
    b_size=[]
    generated_bundles=[]
    for user in sorted(user_bundle_map.keys())[:100]:
        new_bundle = generate_bundle(items_set, user, 4, 1000,size)
        rank = get_bundle_rank(user, new_bundle, bundle_item_map, bundle_diversity_map)
        purchased_bundles = len(user_bundle_map[user])
        aggregate_diversity=aggregate_diversity.union(set(new_bundle))
        generated_bundles.append(new_bundle)
        pred_score.append(rank)
        b_size.append(len(new_bundle)*1.0)
        print 'Rank of user %d : %d, Size of bundle : %d, Bundles purchased : %d Aggregate diversity: %d Score: %f, Average bundle size: %f' %(user, 
                                                                                     rank, 
                                                                                     len(new_bundle),                                 
                                                                                     purchased_bundles,
                                                                                     len(aggregate_diversity),
                                                                                     1.0+np.mean(pred_score),
                                                                                     np.mean(b_size))
    diversities.append(len(aggregate_diversity))
    scores.append(1.0+np.mean(pred_score))
    bundle_sizes.append(np.mean(b_size)*100)

