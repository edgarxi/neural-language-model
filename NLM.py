from __future__ import division
import utility
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
def to_onehot(X, leng): #given a vector of n integers, convert them into a one-hot encoding of 2000xn 
	res = np.zeros((len(X), 8000)) #vocab size x n
	res[range(leng), X] = 1
	return res

class NLM(object):
	def __init__(self, dim = 16, nhidden=128, nwords = 3, vocabSize = 8000, nonlinear=True):
		a1 =  (6.0/(vocabSize+dim))**0.5
		self.C = np.random.uniform(-a1, a1, (vocabSize, dim))
		# need a word vector representer
		self.H = np.random.uniform(-a1, a1, (nwords*dim,nhidden,)) #(48x128)
		self.bias1 = np.zeros(nhidden) #128xbatch size	
		self.U = np.random.uniform(-a1, a1, (nhidden,vocabSize)) #
		self.bias2 = np.zeros(vocabSize)#np.random.uniform(-a1, a1, (vocabSize))
		self.dim = dim
		self.trainperp = []
		self.valperp = []
		self.nonlinear = nonlinear
		self.nhidden = nhidden
	def forwardProp(self, X):
		O = np.dot(X, self.H)+self.bias1 #these are correct i think
		A = O
		if self.nonlinear:
			A = np.tanh(O) # tanh layer
		B = np.dot(A,self.U)+self.bias2
		Y = utility.softmax(B) # 8,000 by 20
		return Y

	def acc(self, X):
		w1 = X[:,0] # batch_size x 1
		w2 = X[:,1] 
		w3 = X[:,2]
		T  = X[:,3]
		T = to_onehot(T, len(X))
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
		return utility.perplexity(Y, T)
	def visualize_(self):
		plt.plot(self.trainperp, label = "train perplexity")
		plt.plot(self.valperp, label = "val perplexity")
		#plt.plot(self.)
		plt.xlabel('epoch')
		plt.title('{} layers without nonlinearity'.format(self.nhidden))
		plt.legend(loc='upper left')
		plt.savefig('{}layers'.format(self.nhidden))



	def train(self, X, Val, epochs=2, batchsize = 20): #where does the indexing go?
		n = len(X)
		for epoch_num in range(epochs):
			print "epoch number: {} ".format(epoch_num)
			mini_batches = [
				X[k:k+batchsize]
				for k in xrange(0, n, batchsize)]
			np.random.shuffle(mini_batches)
			perplexity = []
			for i, mini_batch in enumerate(mini_batches): #make a progress bar 
				if i%(len(mini_batches)//100)==0:
					sys.stdout.write("-")
				perp = self.update_minibatch(mini_batch, lr = 0.001, batchsize= len(mini_batch))
				perplexity.append(perp)

			#loss, perplexity = self.acc(Val)
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
			#print "validation perplexity: {}".format(self.acc(Val))
		np.save("train_{}".format(self.nhidden), self.trainperp)
		np.save("val_{}".format(self.nhidden), self.valperp)
		print "finished training!"
	def update_minibatch(self, X, lr = 0.001, batchsize=20):
	
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
			O = np.dot(e, self.H)+self.bias1 #these are correct i think
			A = O
			if self.nonlinear:
				A = np.tanh(O) # tanh nonlinearity 
			B = np.dot(A,self.U)+self.bias2
			Y = utility.softmax(B) # 8,000 by 20

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
			return perplexity
X = utility.parse("train-new.txt")
Val = utility.parse("val-new.txt")
model = NLM(nonlinear=True)
#model.update_minibatch(X)
print "dimensions OK!"
model.train(X, Val, epochs = 5)
model.visualize_()
