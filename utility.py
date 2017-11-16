import numpy as np 
import pandas as pd
import os
from collections import Counter

# extract 7997 most common words
#df = pd.read_csv('train.txt',sep='\n')
#df = np.array(df)

with open ("train.txt", "r") as myfile:
    data=myfile.read().replace('\n', '').split(' ')

def lookupTable(data):
	freqs = Counter(data)
	#print freqs
	mostCommon = [i[0] for i in freqs.most_common(7997)] #7997 most common words
	mostCommon += ['START', 'END', 'UNK']
	#print mostCommon
	lookupTable = dict(zip(mostCommon, range(8000)))
	return lookupTable

#print lookupTable(data)['UNK']

def clean(fpath):
	with open(fpath, "r") as myfile:
		data = [i.replace('\n', '') for i in myfile.readlines()]
		for i, sentence in enumerate(data):
			data[i] = "START "+sentence+ " END"
		return data

# data = clean('train.txt')

# thefile = open('train1.txt', 'w')
# for item in data:
# 	thefile.write("%s\n" % item)

#plot most common 4-grams... ummmmm