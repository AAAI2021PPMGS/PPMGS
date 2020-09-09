import numpy as np
from scipy.io import loadmat

import sys,getopt

from simulator import SiteSimulator,LONG_MAXIMUM
from secure_ring_summation import SecureRingSummation
from dlr import EDMcompletion,DMR,MultiPerturbPPSC

from sklearn.utils import check_random_state
from sklearn.metrics import f1_score
from sklearn.metrics import euclidean_distances

import time

from G50C_loader import loadG50C_fullsize_balanced_labeled

def test_G50C_DMR(num_labeled=7,num_train=52,num_sites=7,sigma=17.5**2,p1=10,p2=890,step_size=1e-5,max_iter=1500,gammaA=1e-2,gammaI=1e-2,m=48,nn=50,tol=1e-3,edm_rank=48):
	sites=[SiteSimulator() for i in range(num_sites)]
	sins=SecureRingSummation(2,5)
	ppscins=MultiPerturbPPSC()
	ppscins.genR(m,50)
	edmcins=EDMcompletion(ppscins,sins)
	dmrins=DMR(sins,edmcins)
	ds,ls,trd,trl,ted,tel=loadG50C_fullsize_balanced_labeled(num_labeled,num_train,num_sites)
	for i in range(num_sites):
		sites[i].buff['data']=ds[i]
		sites[i].buff['labels']=ls[i]
	edmcins.completion(sites,'data',num_sites*num_train,50,p1,p2,step_size,max_iter,tol,edm_rank)
	edm=euclidean_distances(np.vstack(ds),squared=True)
	#print(np.linalg.matrix_rank(edm))
	diff=sites[0].buff['global_edm']-edm
	edmerror=np.sqrt((diff**2).sum()/(edm**2).sum())

	for i in range(num_sites):
		sites[i].buff['data']=ds[i]
		sites[i].buff['labels']=ls[i]
	dmrins.fit_with_edm(sites,'data','labels','array_num_data',gammaA,gammaI,sigma,nn,2)
	
	dmrins.label(sites,trd,sigma)
	trdpred=sites[0].buff['new_label']
	#trdpred=trdpred/abs(trdpred)
	train_error=np.count_nonzero(trdpred-trl)
	#return trdpred,trl
	train_f1micro=f1_score(trl.flatten(),trdpred.flatten(),average='micro')
	train_f1macro=f1_score(trl.flatten(),trdpred.flatten(),average='macro')

	dmrins.label(sites,ted,sigma)
	tedpred=sites[0].buff['new_label']
	#tedpred=tedpred/abs(tedpred)
	test_error=np.count_nonzero(tedpred-tel)
	test_f1micro=f1_score(tel.flatten(),tedpred.flatten(),average='micro')
	test_f1macro=f1_score(tel.flatten(),tedpred.flatten(),average='macro')

	master_time=sites[0].runtime
	master_count_comm=sites[0].count_send
	master_size_comm=sites[0].size_send
	master_backup_size_comm=sites[0].backup_size_send

	member_time=sum((sites[i].runtime for i in range(1,num_sites)))
	member_count_comm=sum((sites[i].count_send for i in range(1,num_sites)))
	member_size_comm=sum((sites[i].size_send for i in range(1,num_sites)))
	member_backup_size_comm=sum((sites[i].backup_size_send for i in range(1,num_sites)))

	return master_time,master_count_comm,master_backup_size_comm,master_size_comm,member_time,member_count_comm,member_backup_size_comm,member_size_comm,train_f1micro,train_f1macro,train_error,test_f1micro,test_f1macro,test_error,edmerror

if __name__=='__main__':
	opts,args = getopt.getopt(sys.argv[1:],"",['l=','n=','m=','sigma=','p1=','p2','step_size=','max_iter=','gammaA=','gammaI=','m_rpm=','nn=','tol=','edm_rank='])
	l=7
	n=52
	m=7
	sigma=17.5**2
	p1=10
	p2=890
	step_size=1e-5
	max_iter=1500
	gammaA=1e-2
	gammaI=1e-2
	m_rpm=48
	nn=50
	tol=1e-3
	edm_rank=48

	for name,value in opts:
		if name =='--l':
			l=int(value)
		elif name=='--n':
			n=int(value)
		elif name=='--m':
			m=int(value)
		elif name=='--sigma':
			sigma=float(value)
		elif name=='--p1':
			p1=int(value)
		elif name=='--p2':
			p2=int(value)
		elif name=='--step_size':
			step_size=float(value)
		elif name=='--max_iter':
			max_iter=int(value)
		elif name=='--gammaA':
			gammaA=float(value)
		elif name=='--gammaI':
			gammaI=float(value)
		elif name=='--m_rpm':
			m_rpm=int(value)
		elif name=='--nn':
			nn=int(value)
		elif name=='--tol':
			tol=float(value)
		elif name=='--edm_rank':
			edm_rank=int(value) 
	train_size=n*m
	test_size=550-train_size
	print('Performing DLR on G50C')
	print('Settings:')
	print('\t',l,'labeled data per participant')
	print('\t',n,'total training data per participant')
	print('\t',m,'participants;')
	print('\thyper-parameter sigma=',sigma)
	print('\t',p1,'instances to share per participant')
	print('\t',p2,'element of local EDM to share per participant')
	print('\tstep size for EDM completion:',step_size)
	print('\t',max_iter,'iterations of EDM completion')
	print('\tthe tolerance for EDM change:',tol)	
	print('\tassumed EDM rank:',edm_rank)
	print('\tregulariztion parameter gammaA=',gammaA)
	print('\tregulariztion parameter gammaI=',gammaI)
	print('\tthe number of projecting dimension in multicative perturbation:',m_rpm)
	print('\tthe number of nearest neighors in graph',nn)
	result=test_G50C_DMR(l,n,m,sigma,p1,p2,step_size,max_iter,gammaA,gammaI,m_rpm,nn,tol,edm_rank)
	print('Result:')
	print('\tTime Cost Per-participant: ',(result[0]+result[4])/m)
	print('\tData Exchanges Per-participant: ',((result[2]+result[6])*LONG_MAXIMUM+result[3]+result[7])/m)
	print('\tTrain f1-macro: ',result[9])
	print('\tTrain ACC: ',1-result[10]/train_size)
	print('\tTest f1-macro: ',result[12])
	print('\tTest ACC: ',1-result[13]/test_size)
	print('\tEDM completion Error Rate: ',result[-1])