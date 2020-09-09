import numpy as np
from scipy.io import loadmat

import sys,getopt

from simulator import SiteSimulator,LONG_MAXIMUM
from secure_ring_summation import SecureRingSummation
from pssl import Secure2PC,PASSL

from sklearn.utils import check_random_state
from sklearn.metrics import f1_score

import time

from G50C_loader import loadG50C_fullsize_balanced_labeled

def test_G50C_PASSL(num_labeled=7,num_train=52,num_sites=7,sigma=17.5**2,max_iter=20,dmin=0,dmax=150,K=7,R=200,lamb=7,alpha=0.5):
	sites=[SiteSimulator() for i in range(num_sites)]
	sins=SecureRingSummation(2,5)
	s2ins=Secure2PC()
	pins=PASSL(s2ins,sins)
	ds,ls,trd,trl,ted,tel=loadG50C_fullsize_balanced_labeled(num_labeled,num_train,num_sites)
	for i in range(num_sites):
		sites[i].buff['data']=ds[i]
		sites[i].buff['labels']=ls[i]
	pins.graph_construction(sites,'data',K,R,dmin,dmax,lamb,sigma,2)
	pins.distributed_label_propagation(sites,alpha,'labels',max_iter,2)

	trdpred=np.vstack([sites[i].buff['passl_A'] for i in range(num_sites)])
	trdpred=trdpred.argmax(axis=1)+1
	
	#trdpred=trdpred/abs(trdpred)
	train_error=np.count_nonzero(trdpred-trl.flatten())
	#return trdpred,trl
	train_f1micro=f1_score(trl.flatten(),trdpred.flatten(),average='micro')
	train_f1macro=f1_score(trl.flatten(),trdpred.flatten(),average='macro')

	master_time=sites[0].runtime
	master_count_comm=sites[0].count_send
	master_size_comm=sites[0].size_send
	master_backup_size_comm=sites[0].backup_size_send

	member_time=sum((sites[i].runtime for i in range(1,num_sites)))
	member_count_comm=sum((sites[i].count_send for i in range(1,num_sites)))
	member_size_comm=sum((sites[i].size_send for i in range(1,num_sites)))
	member_backup_size_comm=sum((sites[i].backup_size_send for i in range(1,num_sites)))

	return master_time,master_count_comm,master_backup_size_comm,master_size_comm,member_time,member_count_comm,member_backup_size_comm,member_size_comm,train_f1micro,train_f1macro,train_error#,test_f1micro,test_f1macro,test_error


if __name__=='__main__':
	opts,args = getopt.getopt(sys.argv[1:],"",['l=','n=','m=','sigma=','max_iter=','dmin=','dmax=','K=','R=','lamb=','alpha='])
	l=7
	n=52
	m=7
	sigma=17.5**2
	max_iter=20
	dmin=0
	dmax=150
	K=7
	R=200
	lamb=7
	alpha=0.5

	for name,value in opts:
		if name =='--l':
			l=int(value)
		elif name=='--n':
			n=int(value)
		elif name=='--m':
			m=int(value)
		elif name=='--sigma':
			sigma=float(value)
		elif name=='--max_iter':
			max_iter=int(value)
		elif name=='--dmin':
			dmin=float(value)
		elif name=='--dmax':
			dmax=float(value)
		elif name=='--K':
			K=int(value)
		elif name=='--R':
			R=int(value)
		elif name=='--lamb':
			lamb=int(value)
		elif name=='--alpha':
			alpha=float(value)
	train_size=n*m
	test_size=550-train_size
	print('Performing PSSL on G50C')
	print('Settings:')
	print('\t',l,'labeled data per participant')
	print('\t',n,'total training data per participant')
	print('\t',m,'participants;')
	print('\thyper-parameter sigma=',sigma)
	print('\t',max_iter,'iterations of propagation')
	print('\tclosest distances allowed to connect in inter-participant graphs:',dmin)
	print('\tfurthest distances allowed to connect in inter-participant graphs:',dmax)
	print('\t',K,'nearest neighors to connect in local graph')
	print('\t',R,'connections are built between two clusters')
	print('\tthe number of connections per participant for a node is up to',lamb)
	print('\tpropagation hyper-parameter=',alpha)
	result=test_G50C_PASSL(l,n,m,sigma,max_iter,dmin,dmax,K,R,lamb,alpha)
	print('Result:')
	print('\tTime Cost Per-participant: ',(result[0]+result[4])/m)
	print('\tData Exchanges Per-participant: ',((result[2]+result[6])*LONG_MAXIMUM+result[3]+result[7])/m)
	print('\tTrain f1-macro: ',result[9])
	print('\tTrain ACC: ',1-result[10]/train_size)