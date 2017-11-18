import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# vocab = {}
# def buildvocab():
#   global vocab

#   with open ("val.txt") as lines:
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

# process_file("val")

def lookupTable(data): #create lookup table of most frequent word
	freqs = Counter(data)
	#print freqs
	mostCommon = [i[0] for i in freqs.most_common(7997)] #7997 most common words
	mostCommon += ['START', 'END', 'UNK']
	#print mostCommon
	lookupTable = dict(zip(mostCommon, range(8000)))
	#print freqs.most_common(50)
	return lookupTable

def clean(fpath):
	with open(fpath, "r") as myfile:
		data = [i.replace('\n', '').lower() for i in myfile.readlines()]
		for i, sentence in enumerate(data):
			data[i] = data[i].rstrip()
			data[i] = "START "+sentence+ " END"
		data = [s.split(' ') for s in data]
		data = [[item for sublist in data for item in sublist]][0]
		return data

def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

def ngrams(d, start_indices, stop_indices):
	res = []

	for i,j in zip(start_indices, stop_indices):
		nglist = d[i:j+1]
		res.append(find_ngrams(nglist, 4))
	res = [[item for sublist in res for item in sublist]][0] #flatten
	res = [' '.join(i) for i in res]
	res = [i.strip() for i in res]
	return res

def plot_grams(grams):
	#print len(grams)
	freqs = Counter(grams)
	return freqs.most_common(100)
def run():
	return 0

def parse(fpath):
	with open (fpath, "r") as myfile:
		data=myfile.read().replace('\n', ' ').split(' ')

	#print data

	tab = lookupTable(data)
	#print tab
	#data_cleaned = clean('train.txt') #clean 
	
	for ind, word in enumerate(data):
			if word not in tab:	
				data[ind]= 'UNK'

	start_indices = [i_ for i_, x_ in enumerate(data) if x_=="START"]
	stop_indices = [i for i,x in enumerate(data) if x=="END"]
	grams =  ngrams(data, start_indices, stop_indices)
	#print grams
	def words_to_indices(grams):
		res = []
		for sentence in grams:
			sentenceList = []
			for word in sentence.split(' '):
				#print word
				sentenceList.append(tab[word])
			res.append(sentenceList)
		return res
	indices = words_to_indices(grams)
	res =  np.array(indices) #
	#print "shape of the res thing {} ".format(np.shape(res))
	#print res[:10]
	return res
	#print indices[:20]
	#print inds[:20]

	#print plot_grams(grams)
	
#parse("train-new.txt")

def generate_ngram_arrays(ngrams):
	return 0

def tanh(x):
	return np.tanh(x)

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def crossEntropy(a,y): #if we use the cross entropy for a minibatch, how?
	return (-1/len(y))*np.sum(y*np.log(a)+(1-y)*np.log(1-a)) 

def perplexity(a,y):
	return 2**crossEntropy(a,y)