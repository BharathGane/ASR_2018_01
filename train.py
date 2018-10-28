import pandas as pd
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pickle
import os


#Training code
n_components = [2,4,8,16,32,64,128,256]
types = ["mfcc","mfcc_delta","mfcc_delta_delta"]
for k in types: 
	df = pd.read_hdf("./features/"+k+"/timit.hdf")
	features = np.array(df["features"].tolist())
	labels = np.array(df["labels"].tolist())
	# print np.unique(labels)
	temp = {}
	for i in range(len(labels)):
		if labels[i] in temp.keys():
			temp[labels[i]].append(features[i])
		else:
			temp[labels[i]] = [features[i]]
	for j in n_components:
		print "mkdir ./models/mfcc/"+str(j).zfill(3)
		os.system("mkdir ./models/mfcc/"+str(j).zfill(3))	
		for i in temp.keys():
			f = open("./models/mfcc/"+str(j).zfill(3)+"/_"+i+".pkl","wb")
			g = mixture.GaussianMixture(n_components = j)
			gmm = g.fit(temp[i])
			pickle.dump(gmm, f)
			f.close()

