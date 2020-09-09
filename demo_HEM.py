import numpy as np
from scipy.io import loadmat

from simulator import SiteSimulator,LONG_MAXIMUM
from secure_ring_summation import SecureRingSummation
import hem

from sklearn.utils import check_random_state
from sklearn.metrics import f1_score

import time

import sys,getopt

from G50C_loader import loadG50C_seperated_balanced_labeled

def test_G50C_LocalHEM(num_labeled=7,num_train=52,num_sites=7,sigma=17.5**2):
	mls,mus,ls,trd,trl,ted,tel=loadG50C_seperated_balanced_labeled(num_labeled,num_train,num_sites)
	durs=[]
	trpred=[]
	for i in range(num_sites):
		begin=time.time()
		aul,auu=hem.buildGraph(mls[i],mus[i],sigma)
		ulfunc=hem.energyMinimizationDB(mls[i].shape[0],mus[i].shape[0],ls[i],aul,auu)
		durs.append(time.time()-begin)
		ulfunc/=(ulfunc.sum(axis=0)+1e-9)
		ul=ulfunc.argmax(axis=1)
		#print(ls[i].shape,ul.getA().flatten().shape)
		trpred.append(np.hstack((ls[i],ul.getA().flatten())))
	trpred=np.hstack(trpred)
	train_error=np.count_nonzero(trpred-trl)
	train_f1micro=f1_score(trl.flatten(),trpred.flatten(),average='micro')
	train_f1macro=f1_score(trl.flatten(),trpred.flatten(),average='macro')

	return sum(durs),train_f1micro,train_f1macro,train_error

def test_G50C_globalHEM(num_labeled=7,num_train=52,num_sites=7,sigma=17.5**2):
	mls,mus,ls,trd,trl,ted,tel=loadG50C_seperated_balanced_labeled(num_labeled,num_train,num_sites)
	ml=np.vstack(mls)
	mu=np.vstack(mus)
	l=np.hstack(ls)
	tll=[]
	tul=[]
	num_visited=0
	for i in range(num_sites):
		tll.append(trl[num_visited:num_visited+ls[i].shape[0]])
		tul.append(trl[num_visited+ls[i].shape[0]:num_visited+num_train])
		num_visited+=num_train
	#print(trl.shape)
	trl=np.hstack((np.hstack(tll),np.hstack(tul)))
	#print(trl.shape)

	begin=time.time()
	aul,auu=hem.buildGraph(ml,mu,sigma)
	ulfunc=hem.energyMinimizationDB(ml.shape[0],mu.shape[0],l,aul,auu)
	dur=time.time()-begin
	ulfunc/=(ulfunc.sum(axis=0)+1e-9)
	ul=ulfunc.argmax(axis=1)
		#print(ls[i].shape,ul.getA().flatten().shape)
	trpred=np.hstack((l,ul.getA().flatten()))

	train_error=np.count_nonzero(trpred-trl)
	train_f1micro=f1_score(trl.flatten(),trpred.flatten(),average='micro')
	train_f1macro=f1_score(trl.flatten(),trpred.flatten(),average='macro')

	return dur,train_f1micro,train_f1macro,train_error

if __name__=='__main__':
	opts,args = getopt.getopt(sys.argv[1:],"",['local','l=','n=','m=','r=','sigma='])
	l=7
	n=52
	m=7
	sigma=17.5**2
	local=False
	for name,value in opts:
		if name=='--local':
			local=True
		elif name=='--l':
			l=int(value)
		elif name=='--n':
			n=int(value)
		elif name=='--m':
			m=int(value)
		elif name=='--sigma':
			sigma=float(value)
	train_size=n*m
	test_size=550-train_size
	print('Performing PPMGS on G50C')
	print('Settings:')
	if local:
		print('\tLocal mode')
	else:
		print('\tCentral mode')
	print('\t',l,'labeled data per participant')
	print('\t',n,'total training data per participant')
	print('\t',m,'participants')
	print('\thyper-parameter sigma=',sigma)
	if local:
		result=test_G50C_LocalHEM(l,n,m,sigma)
	else:
		result=test_G50C_globalHEM(l,n,m,sigma)
	print('Result:')
	print('\tTime Cost Per-participant: ',result[0]/m)
	
	print('\tTrain f1-macro: ',result[2])
	print('\tTrain ACC: ',1-result[3]/train_size)