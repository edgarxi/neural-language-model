import utility
import numpy as np

def to_onehot(X, leng): #given a vector of n integers, convert them into a one-hot encoding of 2000xn 
	res = np.zeros((8000, len(X))) #vocab size x n
	res[X, range(leng)] = 1
	return res

class NLM(object):
	def __init__(self, dim = 16, nhidden=128, nwords = 3, vocabSize = 8000):
		a1 =  (6.0/(vocabSize+dim))**0.5
		self.C = np.random.uniform(-a1, a1, (vocabSize, dim))
		# need a word vector representer
		self.H = np.zeros((nhidden,nwords*dim)) #(128x48)
		self.bias1 = np.zeros((nhidden,1)) #128xbatch size	
		self.U = np.random.normal(size = (vocabSize, nhidden))
		self.bias2 = np.zeros((vocabSize,1)) 

	def forwardProp(self, X):
		O = np.dot(self.H, X.T)+self.bias1 #these are correct i think #128 x bsize
		#A = np.tanh(O) # tanh layer
		B = np.dot(self.U, O)+self.bias2 #8000x20 
		Y = utility.softmax(B) #8000x20 or 8,000 by vocab 86402 rn
		return Y

	def train(self, X, epochs=100, batchsize = 20): #where does the indexing go?
		n = len(X)
		for epoch_num in range(epochs):
			print "epoch number: {} ".format(epoch_num)
			mini_batches = [
				X[k:k+batchsize]
				for k in xrange(0, n, batchsize)]
			for mini_batch in mini_batches:
				self.update_minibatch(mini_batch)

	def update_minibatch(self, X, lr = 0.001):
			w1 = X[:,0] # batch_size x 1
			w2 = X[:,1] 
			w3 = X[:,2]

			T  = X[:,3] #8,000 x 20 ?
			T = to_onehot(T, len(X))
			c1 = self.C[w1] #continous representation of the word. 
			c2 = self.C[w2] #dimensions: batch_size x depth 
			c3 = self.C[w3]
			#T =  self.C[w4]
			e = np.concatenate((c1, np.concatenate((c2, c3), axis=1)), axis=1) #batch_size x 3D

			# ____________________________________forward propogation step__________________________________
			O = np.dot(self.H, e.T)+self.bias1 #these are correct i think
			#A = np.tanh(O) # tanh layer
			B = np.dot(self.U, O)+self.bias2
			Y = utility.softmax(B) # 8,000 by 20

			# _____________________________________backpropagation step __________________________________
			partial_B = Y-T # 8,000  x 20 
			partial_U = np.dot(partial_B,O.T) # U is (vocabsizex128) #???
			partial_b2 = np.sum(partial_B,axis=0)/batchsize
			partial_O =  np.dot(self.U.T, partial_B) # (128xbatch) 
			partial_H = np.dot(partial_O, e)
			partial_X = np.dot(self.H.T, partial_O) 
			partial_b1 = np.sum(partial_O, axis=0)/batchsize
			
			
			
batchsize = 20

X = utility.parse("train-new.txt")
model = NLM()
#model.update_minibatch(X)
print "dimensions OK"
model.train(X)

