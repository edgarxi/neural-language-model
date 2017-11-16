import utility
import numpy as np

class NLM(object):
	def __init__(self, dim = 16, nhidden=128, nwords = 3, vocabSize = 8000):
		#self.C = np.zeros(())
		self.H = np.zeros((nhidden,nwords*dim)) #(128x16)
		self.bias1 = np.zeros((nhidden,1)) #128xbatch size	
		self.U = np.random.normal(size = (vocabSize, nhidden))
		self.bias2 = np.zeros((vocabSize,1)) 

	def forwardProp(self, X):
		O = np.dot(self.H, X)+self.bias1 #these are correct i think
		#A = np.tanh(O) # tanh layer
		B = np.dot(self.U, O)+self.bias2
		Y = utility.softmax(B)
		return Y

	def train(self, epochs=100):
		return None

batchsize = 20
X = np.zeros((48, batchsize )) #some tests
model = NLM()
Y = model.forwardProp(X)
print np.shape(Y)
