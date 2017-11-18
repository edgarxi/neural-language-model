# vocab ={}

# import torch
# import torch.nn as nn

# def buildvocab():
#   global vocab

#   with open ("train.txt") as lines:
#     for line in lines:
#       line = line.rstrip()
#       words = line.split(" ")
#       for word in words:
#         word = word.lower()
#         vocab[word] = vocab.get(word,0)+1


#   vocab = sorted(vocab.iteritems(), key = lambda (k,v): (v,k), reverse = True)
#   vocab = dict(vocab[:7997])

# buildvocab()


# def process_file(filetype):
#   file = open(filetype+"-new.txt", "w+")
#   with open(filetype+".txt") as lines:
#     for line in lines: 
#       line =line.rstrip()
#       temp = ["START"]
#       words = line.split(" ")
#       for word in words:
#         word = word.lower()
#         if word in vocab:
#           temp.append(word)
#         else:
#           temp.append("UNK")
#       temp.append("END\n")
#       file.write(" ".join(temp))
#   file.close()

# process_file('train')
import numpy as np
import random
# from torch.autograd import Variable
import sys

import matplotlib.pyplot as plt

vocab_size = 8000
vocab = {}
dictionary = {}

r = 0.005

epoch = 100
batch = 128

grams_train = np.empty(0)
grams_validate = np.empty(0)

l_embed = 2
l_hidden = 128

np.random.seed(0)

def parseInput(fname, train):
	f = open(fname, 'r')
	grams = []

	if train:
		cnt = {}
		for line in f.readlines():
			for word in line.lower().split():
				if word in cnt:
					cnt[word] += 1
				else:
					cnt[word] = 1

		toAdd = sorted(cnt.iteritems(), key=lambda (k,v): (v,k), reverse=True)[:vocab_size-3]
		idx = 0
		for (k, v) in toAdd:
			vocab[k] = idx
			idx += 1
	
	for k in vocab:
		dictionary[vocab[k]] = k
	dictionary[vocab_size-3] = 'START'
	dictionary[vocab_size-2] = 'END'
	dictionary[vocab_size-1] = 'UNK'
	mill = [17, 7999, 32, 15]
	gramsCnt = {}
	f = open(fname, 'r')
	for line in f.readlines():
		ids = []
		ids.append(vocab_size-3)
		for word in line.lower().split():
			if word in vocab:
				ids.append(vocab[word])
			else:
				ids.append(vocab_size-1)
		ids.append(vocab_size-2)

		for i in range(len(ids)-3):
			gram = ids[i:i+4]
			grams.append(gram)
			concat = " ".join(dictionary[g] for g in gram)
			if concat in gramsCnt:
				gramsCnt[concat] += 1
			else:
				gramsCnt[concat] = 1
	#print len(grams)
	print (grams.count(mill))
	plotDistribution(gramsCnt)
	#print np.shape(grams)
	#return np.array(grams)

def plotDistribution(gramsCnt):
	top50 = []
	t = []
	for (k, v) in sorted(gramsCnt.iteritems(), key=lambda (k,v): (v,k), reverse=True)[:50]:
		top50.append(v)
		t.append(k)

	# for w in t:
	#   print w
	plt.hist(top50, bins='auto')
	#plt.show()

def softmax(x):
	e_x = np.exp(x)
	return e_x / np.sum(e_x, axis=1)[:,None]

def validate(W3, W2, W1, b3, b2, gamma1, beta1, gamma2, beta2, grams):
	test = []
	test.append(['I', 'like', 'to', 'party'])
	# test.append(['get', 'a', 'different', 'girlfriend'])
	# test.append(['have', 'a', 'good', 'car'])
	# test.append(['I', 'have', 'a', 'bag'])

	ids = []
	for t in test:
		id_now = []
		for w in t:
			if w in vocab:
				id_now.append(vocab[w])
			else:
				id_now.append(vocab_size-1)
		ids.append(id_now)
	ids = np.array(ids)
	w1 = np.append(ids[:,0], grams[:,0]) #what 
	w2 = np.append(ids[:,1], grams[:,1])
	w3 = np.append(ids[:,2], grams[:,2])
	w4 = np.append(ids[:,3], grams[:,3])

	e1 = W1[w1]
	e2 = W1[w2]
	e3 = W1[w3]
	e = np.concatenate((e1, np.concatenate((e2, e3), axis=1)), axis=1)

	h1, cache1 = norm(e.dot(W2) + b2, gamma1, beta1)
	# act1 = np.tanh(h1)
	act1 = h1
	h2, cache2 = norm(act1.dot(W3) + b3, gamma2, beta2)
	s = softmax(h2)

	# for i in range(4):
		# print s[i][np.argmax(s[i])]
		# print dictionary[np.argmax(s[i])]

	loss = -np.sum(np.log(s[np.arange(w1.shape[0]), w4]))
	loss /= w1.shape[0]
	
	perplexity = -np.sum(np.log2(s[np.arange(w1.shape[0]), w4]))
	perplexity /= w1.shape[0]
	perplexity = np.exp2(perplexity)
	return loss, perplexity

def norm(x, gamma, beta, eps = 0.0001):
	return x, None
	N, D = x.shape
	deltax = x - 1./N * np.sum(x, axis = 0)
	var = 1./N * np.sum(deltax ** 2, axis = 0)
	sqrtvar = np.sqrt(var + eps)
	ivar = 1./sqrtvar
	xout = deltax * ivar
	out = gamma * xout + beta
	cache = (xout,gamma,deltax,ivar,sqrtvar,var,eps)
	return out, cache

def backnorm(dout, cache):
	return dout, 0, 0
	xout,gamma,deltax,ivar,sqrtvar,var,eps = cache
	N,D = dout.shape
	dbeta = np.sum(dout, axis=0) / N
	dgamma = np.sum(dout*xout, axis=0) / N
	dxout = dout * gamma
	dsqrtvar = -1. /(sqrtvar**2) * np.sum(dxout*deltax, axis=0)
	dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
	dsq = 1. /N * np.ones((N,D)) * dvar
	dx1 = (dxout * ivar + 2 * deltax * dsq)
	dmu = -1 * np.sum(dx1, axis=0)
	dx2 = 1. /N * np.ones((N,D)) * dmu
	dx = (dx1 + dx2)
	return dx, dgamma, dbeta

def train(): #where is the data being passed in? 
	a1 = (6.0/(vocab_size+l_embed))**0.5
	W1 = np.random.uniform(-a1, a1, (vocab_size, l_embed))

	a2 = (6.0/(l_hidden+3*l_embed))**0.5
	W2 = np.random.uniform(-a2, a2, (3*l_embed, l_hidden))
	b2 = np.zeros((1, l_hidden))

	a3 = (6.0/(l_hidden+vocab_size))**0.5
	W3 = np.random.uniform(-a3, a3, (l_hidden, vocab_size))
	b3 = np.zeros((1, vocab_size))

	gamma1 = 1.
	beta1 = 0.
	gamma2 = 1.
	beta2 = 0.

	idx = range(grams_train.shape[0])
	train_loss = []
	valid_loss = []
	valid_perp = []
	for cur_epoch in range(epoch):
		random.shuffle(idx)
		total_sum = 0.0
		for i in range(grams_train.shape[0]/batch):
			gram = grams_train[idx[i*batch:(i+1)*batch]]

			w1 = gram[:,0]
			w2 = gram[:,1]
			w3 = gram[:,2]
			w4 = gram[:,3]
			
			e1 = W1[w1] #continuous representations of words
			e2 = W1[w2] 
			e3 = W1[w3]
			e = np.concatenate((e1, np.concatenate((e2, e3), axis=1)), axis=1) # e becomes batch_size x 3d 

			h1, cache1 = norm(e.dot(W2) + b2, gamma1, beta1)
			# act1 = np.tanh(h1)
			act1 = h1
			h2, cache2 = norm(act1.dot(W3) + b3, gamma2, beta2)
			s = softmax(h2)

			total_sum += -np.sum(np.log(s[np.arange(w1.shape[0]), w4]))

			delta_h2 = s
			delta_h2[np.arange(s.shape[0]), w4] -= 1.0
			delta_h2, delta_gamma2, delta_beta2 = backnorm(delta_h2, cache2)

			delta_w3 = act1.T.dot(delta_h2)
			delta_b3 = np.sum(delta_h2, axis=0)

			# delta_h1 = delta_h2.dot(W3.T)*(1-act1**2)
			delta_h1 = delta_h2.dot(W3.T)
			delta_h1, delta_gamma1, delta_beta1 = backnorm(delta_h1, cache1) #batchnorm
			delta_w2 = e.T.dot(delta_h1)
			delta_b2 = np.sum(delta_h1, axis=0)

			delta_e = delta_h1.dot(W2.T)
			W1[w1] -= r * delta_e[:,:l_embed]
			W1[w2] -= r * delta_e[:,l_embed:2*l_embed]
			W1[w3] -= r * delta_e[:,2*l_embed:3*l_embed]

			W3 -= r * delta_w3 / batch 
			W2 -= r * delta_w2 / batch 
			b3 -= r * delta_b3 / batch 
			b2 -= r * delta_b2 / batch 
			gamma1 -= r * delta_gamma1
			beta1 -= r * delta_beta1
			gamma2 -= r * delta_gamma2
			beta2 -= r * delta_beta2

		train_loss.append(total_sum / grams_train.shape[0]) 
		loss, perplexity = validate(W3, W2, W1, b3, b2, gamma1, beta1, gamma2, beta2, grams_validate)
		valid_loss.append(loss) 
		valid_perp.append(perplexity) 

		print cur_epoch
		print train_loss[-1]
		print valid_loss[-1]
		print valid_perp[-1]

	
	idx = range(vocab_size)
	random.shuffle(idx)
	x = []
	y = []
	for i in range(500):
		x.append(W1[idx[i]][0])
		y.append(W1[idx[i]][1])

	plt.scatter(x, y)
	plt.show()
		
	# print train_loss
	# print valid_loss
	# print valid_perp


# for i in range(1010,1020):
	# for w in grams_train[i]:
		# print dictionary[w]
	# print "-------------"
	# for w in grams_validate[i]:
		# print dictionary[w]
	# print "-------------"

# def gramsToTensors(grams):
	# grams_hot = np.zeros((grams.shape[1], grams.shape[0], vocab_size), dtype=np.float32)
	# for i in range(grams.shape[0]):
		# for j in range(grams.shape[1]):
			# grams_hot[j][i][grams[i][j]] = 1.0
	# tensors = []
	# for i in range(3):
		# tensors.append(torch.from_numpy(grams_hot[i]))
	# return tensors, torch.from_numpy(grams[:,3])

# class RNN(nn.Module):
	# def __init__(self):
		# super(RNN, self).__init__()

		# self.i2e = nn.Linear(vocab_size, l_embed)
		# self.e2h = nn.Linear(l_embed, l_hidden)
		# self.h2h = nn.Linear(l_hidden, l_hidden)
		# self.batchnorm_h = nn.BatchNorm1d(l_hidden)
		# self.tanh = nn.Tanh()
		# self.h2s = nn.Linear(l_hidden, vocab_size)
		# self.batchnorm_s = nn.BatchNorm1d(vocab_size)
		# self.softmax = nn.LogSoftmax()

	# def forward(self, input, hidden):
		# embed = self.i2e(input)
		# hidden = self.tanh(self.batchnorm_h(torch.add(self.e2h(embed), self.h2h(hidden))))
		# s = self.batchnorm_s(self.h2s(hidden))
		# output = self.softmax(s)
		# return output, hidden

	# def initHidden(self):
		# return Variable(torch.zeros(batch, l_hidden))

# def trainRnnBatch(rnn, criterion, tensors, category_tensor):
	# hidden = rnn.initHidden()
	# rnn.zero_grad()

	# for i in range(3):
		# if i == 1 and np.random.uniform() < 1.0:
			# hidden.detach_()
		# output, hidden = rnn(Variable(tensors[i]), hidden)


	# loss = criterion(output, Variable(category_tensor))
	# loss.backward()

	# for p in rnn.parameters():
		# p.data.add_(-r, p.grad.data)

	# return output, loss.data[0]

# def validate_rnn(rnn, criterion):
	# hidden = Variable(torch.zeros(grams_validate.shape[0], l_hidden))
	# rnn.zero_grad()

	# tensors, category_tensor = gramsToTensors(grams_validate)
	# for i in range(3):
			# output, hidden = rnn(Variable(tensors[i]), hidden)
	# loss = criterion(output, Variable(category_tensor))
	# return output, loss

# def trainRnn():
	# rnn = RNN()
	# criterion = nn.CrossEntropyLoss()
	# idx = range(grams_train.shape[0])
	# for _ in range(epoch):
		# random.shuffle(idx)
		# for i in range(grams_train.shape[0]/batch):
			# grams = grams_train[idx[i*batch:(i+1)*batch]]
			# tensors, category_tensor = gramsToTensors(grams)
			# output, loss = trainRnnBatch(rnn, criterion, tensors, category_tensor)
			# if i % 10 == 9:
				# output, loss = validate_rnn(rnn, criterion)
				# print loss

grams_train = parseInput("train.txt", True)
grams_validate = parseInput("val.txt", False)
train()
# plotDistribution()
