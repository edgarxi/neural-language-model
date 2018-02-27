from __future__ import division
import pickle
import utility
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
def to_onehot(X, leng): #given a vector of n integers, convert them into a one-hot encoding of 2000xn
	res = np.zeros((len(X), 8000),dtype = np.float32) #vocab size x n
	res[range(leng), X] = 1
	return res

class NLM(object):
	def __init__(self, tab, X, dim = 2, nhidden=128, nwords = 3, vocabSize = 8000, nonlinear=True, generate = False):
		# change everything to single precision
		self.X = X
		a1 =  (6.0/(vocabSize+dim))**0.5
		self.C = np.random.uniform(-a1, a1, (vocabSize, dim)).astype(np.float32)
		# need a word vector representer
		self.H = np.random.uniform(-a1, a1, (nwords*dim,nhidden,)).astype(np.float32) #(48x128)
		self.bias1 = np.zeros(nhidden, dtype = np.float32) #128xbatch size
		self.U = np.random.uniform(-a1, a1, (nhidden,vocabSize)).astype(np.float32) #
		self.bias2 = np.zeros(vocabSize, dtype = np.float32)#np.random.uniform(-a1, a1, (vocabSize))
		self.dim = dim
		self.trainperp = []
		self.valperp = []
		self.nonlinear = nonlinear
		self.nhidden = nhidden
		self.tab = tab
		print "initialized neural language model with {} hidden layers".format(nhidden)
	def forwardProp(self, X):
		O = np.dot(X, self.H)+self.bias1 #these are correct i think
		A = O
		if self.nonlinear:
			A = np.tanh(O) # tanh layer
		#B = np.dot(A,self.U)+self.bias2
		Y = utility.softmax(np.dot(A,self.U)+self.bias2) # 8,000 by 20
		return Y

	def acc(self, X):
		w1 = X[:,0] # batch_size x 1
		w2 = X[:,1]
		w3 = X[:,2]
		T  = X[:,3]
		T = to_onehot(T, len(X))
		#print T
		#print len(T)
		c1 = self.C[w1] #continous representation of the word.
		c2 = self.C[w2] #dimensions: batch_size x depth
		c3 = self.C[w3]
		e = np.concatenate((c1, np.concatenate((c2, c3), axis=1)), axis=1) #batch_size x 3D generated
		# ____________________________________forward propogation step__________________________________
		# O = np.dot(self.H, e.T)+self.bias1 #these are correct i think
		# #A = np.tanh(O) # tanh layer
		# B = np.dot(self.U, O)+self.bias2
		# Y = utility.softmax(B) # 8,000 by 20
		Y = self.forwardProp(e)
		#print len(X)
		p = utility.perplexity(Y,T)
		#print np.shape(p)
		return p
		return utility.perplexity(Y, T)
	def visualize_(self):
		plt.plot(self.trainperp, label = "train perplexity")
		plt.plot(self.valperp, label = "val perplexity")
		#plt.plot(self.)
		plt.xlabel('epoch')
		plt.title('{} layers without nonlinearity'.format(self.nhidden))
		plt.legend(loc='upper left')
		plt.savefig('{}layers'.format(self.nhidden))

	def train(self, X, Val, epochs=2, batchsize = 64, lr = 0.01, debug = False): #where does the indexing go?
		print "learning rate:{}".format(lr)
		self.predict()
		n = len(X)
		#print X
		for epoch_num in range(epochs):
			if debug:
				start = time.time()
			print "epoch number: {} ".format(epoch_num)
			mini_batches = [
				X[k:k+batchsize]
				for k in xrange(0, n, batchsize)]
			np.random.shuffle(mini_batches)
			perplexity = []
			for i, mini_batch in enumerate(mini_batches): #make a progress bar
				# if i%(len(mini_batches)//100)==0:
				# 	sys.stdout.write("-")
				perp = self.update_minibatch(mini_batch, lr, batchsize= batchsize)
				#print perp
				perplexity.append(perp)

			#loss, perplexity = self.acc(Val)
			if debug:
				print "forward pass took {} seconds".format(time.time()-start)
				start = time.time()
			print "calculating perplexities"
			#trainloss = self.acc(X)
			valloss = self.acc(Val)
			#self.trainperp.append(trainloss)
			avg_perp = sum(perplexity)/len(perplexity)
			self.trainperp.append(avg_perp)
			print "calculating validation loss:"
			self.valperp.append(valloss)
			print "train loss:", avg_perp
			print "val loss:", valloss
			#print "crossentropy ", utility.crossEntropy()
			#print "validation perplexity: {}".format(self.acc(Val))
			if debug:
				print "accuracy calculation took {} seconds".format(time.time()-start)
			self.predict()
		np.save("train_{}".format(self.nhidden), self.trainperp)

		np.save("val_{}".format(self.nhidden), self.valperp)
		print "finished training!"
	def distance(self, X,Y):
		return np.linalg.norm(X-Y)

	def predict(self):
		sentences = ["government of the", "city of new", "is going to", "president and chief", "the company said", "man is never", "and chief ,"]
		# with open ("train.txt", "r") as myfile:
		# 	data=myfile.read().replace('\n', ' ').split(' ')

		# wordlist, tab = utility.lookupTable(data)
		#wordlist = np.array(wordlist)
		X = []
		for sentence in sentences:
			sentenceList = []
			for word in sentence.split(' '):
				#print word
				sentenceList.append(self.tab[word])
			X.append(sentenceList)
		#print X
		X = np.array(X)
		#print np.shape(X)
		#print sentenceList
		def word_to_vec(word):
			return self.C[self.tab[word]]

		print "similarity score between New  and york: ", self.distance(word_to_vec("new"), word_to_vec("york"))
		print "similarity score between legislation and bad", self.distance(word_to_vec("legislation"), word_to_vec("bad"))
		#print X
		w1 = X[:,0] # batch_size x 1
		#print w1
		w2 = X[:,1]
		w3 = X[:,2]
		c1 = self.C[w1]#.reshape(1,-1) #continous representation of the word.
		#print c1
		c2 = self.C[w2]#.reshape(1,-1) #dimensions: batch_size x depth
		c3 = self.C[w3]#.reshape(1,-1)
		#print np.shape(c3)
		def concat(a,b,c):
			return np.concatenate((a, np.concatenate((b, c), axis=1)), axis=1)
		for i in range(10):
			e = concat(c1,c2,c3) #batch_size x 3D generated
			Y_pred = self.forwardProp(e)

		#print np.shape(Y_pred)
			Y = (np.argmax(Y_pred, axis=1)).astype(int)
			j = self.C[Y]
			c1, c2, c3 = c2, c3, j
			#print np.shape(c1), np.shape(Y)
			#print X
			print np.array(self.X)[Y]
		#print Y
		for i, word in enumerate(sentences):
			print word
		#return wordlist[Y]

	def update_minibatch(self, X, lr = 0.005, batchsize=20):

			w1 = X[:,0] # batch_size x 1
			w2 = X[:,1]
			w3 = X[:,2]
			w4  = X[:,3] #8,000 x 20 ?
			T = to_onehot(w4, len(X)) # the onehot has COLUMNS. ()8,000 x 20
			c1 = self.C[w1] #continous representation of the word.
			c2 = self.C[w2] #dimensions: batch_size x depth  =
			c3 = self.C[w3] #

			#T =  self.C[w4]
			e = np.concatenate((c1, np.concatenate((c2, c3), axis=1)), axis=1) #batch_size x 3D generated
			#e = e
			#print np.shape(e), np.shape(self.H)
			# ____________________________________forward propogation step__________________________________
			#print np.shape(T)
			O = np.dot(e, self.H)+self.bias1 #these are correct i think
			A = O
			if self.nonlinear:
				A = np.tanh(O) # tanh nonlinearity
			B = np.dot(A,self.U)+self.bias2
			Y = utility.softmax(B) # 8,000 by 20
			#print np.shape(Y)
			# else:
			# 	Y = utility.softmax(np.dot((np.dot(e, self.H)+self.bias1),self.U)+self.bias2)
  			perplexity= utility.perplexity(Y, T)
  			#print perplexity
			#utility.crossEntropy(Y, T)/batchsize
			#print "perplexity: {}".format(loss)
			# _____________________________________backpropagation step __________________________________

			partial_B = (Y-T) # 8,000  x 20
			#print np.shape(partial_B)
			partial_U = np.dot(A.T,partial_B)  #this should be right
			partial_b2 = np.sum(partial_B,axis=0)
			deltaA = 1
			if self.nonlinear:
				deltaA = (1-A**2)
			partial_O =  np.dot(partial_B,self.U.T)*A # (128xbatch)
			partial_H = np.dot(e.T, partial_O)
			partial_X = np.dot(partial_O, self.H.T )  #???????? vocab X 16
			partial_b1 = np.sum(partial_O, axis=0)

			#--------------parameter update step . go with easy step first? ----------------
			self.H -= lr * partial_H / batchsize

			#print np.shape(self.bias1), np.shape(partial_b1)
			self.bias1 -= lr*partial_b1  / batchsize
			self.U -= lr*partial_U / batchsize
			self.bias2 -= lr* partial_b2 /batchsize
			self.C[w1] -= lr*partial_X[:,:self.dim]/ batchsize
			self.C[w2] -= lr*partial_X[:,self.dim:2*self.dim]/ batchsize#[:,self.dim:2*self.dim]
			self.C[w3] -= lr*partial_X[:,2*self.dim:3*self.dim] / batchsize#[:,2*self.dim:3*self.dim]
			#print perplexity
			return perplexity

	def visualize_weights(self):

		a = np.random.random_integers(0, 7999, 500)
		c = self.C[a]
		for i in c:
			plt.scatter(i[0],i[1])
			#ax = plt.subplot(10,10,i)
			#ax.imshow(weights[i-1].reshape(28,28),cmap='gray')
		plt.show()
		plt.savefig('embeddings')

tabx, X, res = utility.parse("train-new.txt")
taby, Val, res1 = utility.parse("val-new.txt")
model = NLM(tabx, X, nhidden = 128, nonlinear=False)
#model.update_minibatch(X)

model.train(res, res1, epochs = 12, lr = 0.5, debug = True)
model.visualize_weights()
# with open('model.pickle', 'wb') as handle:
#     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.visualize_()
