import time
import math
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as KM
#import dataSim
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky as compute_precision_cholesky
from scipy.stats import multivariate_normal as mvn

from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms

from sklearn.neighbors import NearestNeighbors as NN

from scipy import sparse as sp

print('import tag:','Mon')

_TP_=1e-50 #tiny positive

'''
Part 1 Functions
'''

def buildGraph(MatL,MatU,rbf_sigma=None,knn=0):
	num_labeled=MatL.shape[0]
	num_unlabeled=MatU.shape[0]
	data_dim=MatL.shape[1]
	normUU=row_norms(MatU,squared=True)[:,np.newaxis]
	affinity_UL = -0.5*euclidean_distances(MatU,MatL,squared=True,X_norm_squared=normUU)
	affinity_UU = -0.5*euclidean_distances(MatU,squared=True,X_norm_squared=normUU)
	kcomp=num_labeled+num_unlabeled-knn
	affinity_UL/=rbf_sigma
	affinity_UL=np.exp(affinity_UL)
	for i in range(num_unlabeled):
		uslice=affinity_UU[i,i+1:]
		uslice/=rbf_sigma
		uslice=np.exp(uslice)
		affinity_UU[i+1:,i]=uslice
		affinity_UU[i,i+1:]=uslice
		affinity_UU[i,i]=0.0
		if knn>0:
			affinity_i=np.hstack((affinity_UL[i],affinity_UU[i]))
			inds=np.argpartition(affinity_i,kcomp-1)[:kcomp]
			affinity_i=np.ones(num_labeled+num_unlabeled)
			affinity_i[inds]=0.0
			affinity_UL[i]=affinity_i[:num_labeled]
			affinity_UU[i]=affinity_i[num_labeled:]
		row_sum=sum(affinity_UL[i])+sum(affinity_UU[i])
		if row_sum!=0:
			affinity_UL[i]/=row_sum
			affinity_UU[i]/=row_sum
	return affinity_UL,affinity_UU

def buildGraphSK(MatL,MatU,rbf_sigma=None,knn=0):
	nins=NN(knn,None,metric='euclidean').fit(np.vstack((MatL,MatU)))
	W=nins.kneighbors_graph(nins._fit_X,knn,mode='distance')
	W.data=np.exp(-W.data/rbf_sigma)
	affinity_UL=W[MatL.shape[0]:,:MatL.shape[0]]#.toarray()
	affinity_UU=W[MatL.shape[0]:,MatL.shape[0]:]#.toarray()
	return affinity_UL,affinity_UU


def labelPropagationDB(num_labeled, num_unlabeled, labels, affinity_UL, affinity_UU, max_iter = 500, tol = 1e-3):
	num_classes =  max(labels.astype(np.int32))+1#len(np.unique(labels))
	clamp_labels=np.zeros([num_labeled,num_classes],np.float32)
	for i in range(num_labeled):
		clamp_labels[i][int(labels[i])] = 1.0
	label_function=-1*np.ones([num_unlabeled,num_classes],np.float32)
	iterator=0
	changed=tol+1
	pre_label_function = np.zeros((num_unlabeled, num_classes), np.float32)
	while iterator<max_iter and changed>tol:
		pre_label_function = label_function
		iterator+=1
		label_function=np.dot(affinity_UU, label_function)+np.dot(affinity_UL,clamp_labels)
		changed = np.abs(pre_label_function - label_function).sum()
	return label_function

def energyMinimizationDB(num_labeled, num_unlabeled, labels, affinity_UL, affinity_UU):
	num_classes = max(labels.astype(np.int32))+1#len(np.unique(labels))
	clamp_labels=np.zeros([num_labeled,num_classes],np.float32)
	for i in range(num_labeled):
		clamp_labels[i][int(labels[i])] = 1.0
	label_function=np.dot(np.dot(np.matrix(np.eye(num_unlabeled)+1e-10-affinity_UU).I,affinity_UL),clamp_labels)
	return label_function

def energyMinimizationSK(num_labeled, num_unlabeled, labels, DeltaUL, DeltaUU):
	num_classes = max(labels.astype(np.int32))+1#len(np.unique(labels))
	clamp_labels=np.zeros([num_labeled,num_classes],np.float32)
	for i in range(num_labeled):
		clamp_labels[i][int(labels[i])] = 1.0
	DUU=DeltaUL.sum(axis=1)+DeltaUU.sum(axis=1)
	DeltaUU[np.diag_indices_from(DeltaUU)]=DeltaUU.diagonal()-DUU.getA().flatten()-1e-10
	if sp.issparse(DeltaUU):
		DeltaUU=DeltaUU.toarray()
	DeltaUU=np.linalg.inv(DeltaUU)
	DeltaUL=DeltaUL.dot(clamp_labels)

	label_function=-np.dot(DeltaUU,DeltaUL)
	return label_function
