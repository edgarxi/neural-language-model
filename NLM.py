import utility
import numpy as np
import matplotlib.pyplot as plt
def to_onehot(X, leng): #given a vector of n integers, convert them into a one-hot encoding of 2000xn 
	res = np.zeros((8000, len(X))) #vocab size x n
	res[X, range(leng)] = 1
	return res

class NLM(object):
	def __init__(self, dim = 16, nhidden=128, nwords = 3, vocabSize = 8000):
		a1 =  (6.0/(vocabSize+dim))**0.5
		self.C = np.random.uniform(-a1, a1, (vocabSize, dim))
		# need a word vector representer
		self.H = np.random.uniform(-a1, a1, (nhidden,nwords*dim)) #(128x48)
		self.bias1 = np.random.uniform(-a1, a1, (nhidden,1)) #128xbatch size	
		self.U = np.random.uniform(-a1, a1, (vocabSize, nhidden))
		self.bias2 = np.random.uniform(-a1, a1, (vocabSize,1))
		self.dim = dim
	def forwardProp(self, X):
		O = np.dot(self.H, X.T)+self.bias1 #these are correct i think #128 x bsize
		#A = np.tanh(O) # tanh layer
		B = np.dot(self.U, O)+self.bias2 #8000x20 
		Y = utility.softmax(B) #8000x20 or 8,000 by vocab 86402 rn
		return Y

	def acc(self, X):
		w1 = X[:,0] # batch_size x 1
		w2 = X[:,1] 
		w3 = X[:,2]
		T  = X[:,3]

		T = to_onehot(T, len(X))
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
		print len(X)
		return utility.perplexity(Y, T)/len(X)

	def train(self, X, Val, epochs=100, batchsize = 20): #where does the indexing go?
		n = len(X)
		for epoch_num in range(epochs):
			print "epoch number: {} ".format(epoch_num)
			mini_batches = [
				X[k:k+batchsize]
				for k in xrange(0, n, batchsize)]
			for mini_batch in mini_batches:
				self.update_minibatch(mini_batch, batchsize)
			#print "validation perplexity: {}".format(self.acc(Val))

	def update_minibatch(self, X, lr = 0.001, batchsize=20):
			w1 = X[:,0] # batch_size x 1
			w2 = X[:,1] 
			w3 = X[:,2]
			W_t = X[:,:2]

			T  = X[:,3] #8,000 x 20 ?
			T = to_onehot(T, len(X)) # is oneHot wrong?
			c1 = self.C[w1] #continous representation of the word. 
			c2 = self.C[w2] #dimensions: batch_size x depth 
			c3 = self.C[w3]
			#print np.shape(c3)
			#T =  self.C[w4]
			e = np.concatenate((c1, np.concatenate((c2, c3), axis=1)), axis=1) #batch_size x 3D generated 
			print np.shape(e)
			# ____________________________________forward propogation step__________________________________
			O = np.dot(self.H, e.T)+self.bias1 #these are correct i think
			#A = np.tanh(O) # tanh layer
			B = np.dot(self.U, O)+self.bias2
			Y = utility.softmax(B) # 8,000 by 20

			#Y = self.forwardProp(e)

			perplexity = utility.crossEntropy(Y, T)/batchsize
			print "perplexity: {}".format(perplexity)

			# _____________________________________backpropagation step __________________________________
			partial_B = (Y-T) # 8,000  x 20 
			#print np.shape(partial_B)
			partial_U = np.dot(partial_B,O.T) # U is (vocabsizex128) #??? this seems correct. 
			#assert np.shape(partial_U) == np.shape(self.U)
			#print np.shape(self.bias2)
			partial_b2 = np.mean(partial_B,axis=1) #(batchsize,) #bias is 128 by batch size
			
			partial_b2 = partial_b2.reshape(-1, 1)
			#print np.shape(partial_b2)
			assert np.shape(partial_b2) == np.shape(self.bias2)

			#print np.shape(partial_b2)
			partial_O =  np.dot(self.U.T, partial_B) # (128xbatch) 
			assert np.shape(partial_O) == np.shape(O)
			partial_H = np.dot(partial_O, e)
			assert np.shape(partial_H) == np.shape(self.H)
			#print np.shape(partial_H)
			#print np.shape(self.H), np.shape(partial_O)
			partial_X = np.dot(self.H.T, partial_O)  #????????

			print np.shape(partial_X), np.shape(e)
			assert np.shape(partial_X) == np.shape(T)
			partial_b1 = np.sum(partial_O, axis=1).reshape(-1, 1)
			#print np.shape(partial_b1)
			#print np.shape(self.bias1)

			#--------------parameter update step . go with easy step first? ----------------
			self.H -= lr * partial_H / batchsize
			self.bias1 -= lr*partial_b1 / batchsize
			self.U -= lr*partial_U / batchsize
			self.bias2 -= lr* partial_b2# / batchsize
			#print "one:" , np.shape(self.C[w1])
			#print np.shape(partial_X[:self.dim])
			self.C[w1] -= lr*partial_X[:,:self.dim] 
			self.C[w2] -= lr*partial_X[:,self.dim:2*self.dim] 
			self.C[w3] -= lr*partial_X[:,2*self.dim:3*self.dim] 

X = utility.parse("train-new.txt")
Val = utility.parse("val-new.txt")
model = NLM()
#model.update_minibatch(X)
print "dimensions OK!"
model.train(X, Val)

