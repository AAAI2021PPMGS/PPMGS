'''
An implementation of the semi-supervised DPPDM protocol, DLR

citation:
@Article{Fierimonte2017,
  author    = {Roberto Fierimonte and Simone Scardapane and Aurelio Uncini and Massimo Panella},
  title     = {Fully Decentralized Semi-supervised Learning via Privacy-preserving Matrix Completion},
  journal   = {{IEEE} Trans. Neural Networks Learn. Syst.},
  year      = {2017},
  volume    = {28},
  number    = {11},
  pages     = {2699--2711},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl    = {https://dblp.org/rec/journals/tnn/FierimonteSUP17.bib},
  doi       = {10.1109/TNNLS.2016.2597444},
}
'''

import numpy as np
import time

from sklearn.metrics import euclidean_distances

from simulator import DistributedProtocol,TINY_POSITIVE

'''
Sub-Protocol: Euclidean-distance Matrix Completion
'''

'''Protocol Class'''
class EDMcompletion(DistributedProtocol):
	def __init__(self,DistanceComputationProtocol,SecureSummationProtocol):
		DistributedProtocol.__init__(self)
		self.DCP=DistanceComputationProtocol
		self.SSP=SecureSummationProtocol

	def generate_shared_data_array(self,part_sites,dim_data):
		shared_data_array=[]
		shared_data_index=[]
		for s in part_sites:
			s.buff['shared_data_index']=shared_data_index
			s.buff['shared_data_array']=shared_data_array

	def completion(self,part_sites,data_index,num_data,num_dim,p1,p2,step_size,max_iter,tol,edm_rank):
		num_sites=len(part_sites)
		for i in range(num_sites):
			part_sites[i].opt(edm_count_data_partial,[data_index],['local_num_data','array_num_data','global_num_data',i,num_sites])		
		for i in range(num_sites):
			part_sites[i].send(part_sites[:i]+part_sites[i+1:],['temp_1'],['local_num_data'])
			for j in range(num_sites):
				if not i==j:
					part_sites[j].opt(edm_save_local_num_data_partial,['array_num_data','temp_1'],['global_num_data'])
		self.generate_shared_data_array(part_sites,part_sites[0].buff[data_index].shape[1])
		for i in range(num_sites):
			part_sites[i].opt(edm_share_data_partial,[data_index,'array_num_data'],['shared_data_index','shared_data_array',p1,'temp_1',i,self.DCP])
			part_sites[i].send(part_sites[:i]+part_sites[i+1:],['temp_1'],['temp_1'])
		#return
		for i in range(num_sites):
			'''
			attention
			'''
			part_sites[i].opt(edm_new_shared_edm_entrences_partial,[data_index,'array_num_data','shared_data_index','shared_data_array'],['local_edm',i,p2,'temp_1',self.DCP,'local_entrence','change',tol,'array_num_data'])
		for i in range(num_sites):
			part_sites[i].send(part_sites[:i]+part_sites[i+1:],['temp_2'],['temp_1'])
			for j in range(num_sites):
				if not i==j:
					part_sites[j].opt(edm_absorb_edm_entrences_partial,['temp_2','local_edm','local_entrence'])

		V=np.random.random((num_data,edm_rank))
		for s in part_sites:
			s.buff['V']=V
		ni=0
		change=1e8
		if max_iter==0:
			for s in part_sites:
				s.buff['global_edm']=s.buff['local_edm']
			return
		for n in range(max_iter):
			for s in part_sites:
				s.opt(edm_diffusion_gradient_descent_partial,['local_edm','local_entrence','V'],['Vk_tide',step_size,num_sites])
			self.SSP.secure_sum(part_sites,'Vk_tide',part_sites,'V')
			last_change=change
			change=abs(V-part_sites[0].buff['V']).sum()/V.sum()
			#if n>max_iter-15:
				#print(change)
			if change<tol:# or change>last_change:
				break
			V=part_sites[0].buff['V']
			ni=n
		#print(ni)
		for s in part_sites:
			s.opt(edm_final_completion_partial,['V'],['global_edm'])

	def test_completion(self,part_sites,data_index,num_data,num_dim,p1,p2,step_size,max_iter,tol):
		num_sites=len(part_sites)
		for i in range(num_sites):
			part_sites[i].opt(edm_count_data_partial,[data_index],['local_num_data','array_num_data','global_num_data',i,num_sites])		
		for i in range(num_sites):
			part_sites[i].send(part_sites[:i]+part_sites[i+1:],['temp_1'],['local_num_data'])
			for j in range(num_sites):
				if not i==j:
					part_sites[j].opt(edm_save_local_num_data_partial,['array_num_data','temp_1'],['global_num_data'])
		self.generate_shared_data_array(part_sites,part_sites[0].buff[data_index].shape[1])
		for i in range(num_sites):
			part_sites[i].opt(edm_share_data_partial,[data_index,'array_num_data'],['shared_data_index','shared_data_array',p1,'temp_1',i,self.DCP])
			part_sites[i].send(part_sites[:i]+part_sites[i+1:],['temp_1'],['temp_1'])
		for i in range(num_sites):
			'''
			attention
			'''
			part_sites[i].opt(edm_new_shared_edm_entrences_partial,[data_index,'array_num_data','shared_data_index','shared_data_array'],['local_edm',i,p2,'temp_1',self.DCP,'local_entrence','change',tol,'array_num_data'])
		for i in range(num_sites):
			part_sites[i].send(part_sites[:i]+part_sites[i+1:],['temp_2'],['temp_1'])
			for j in range(num_sites):
				if not i==j:
					part_sites[j].opt(edm_absorb_edm_entrences_partial,['temp_2','local_edm','local_entrence'])

		V=np.random.random((num_data,num_dim))
		for s in part_sites:
			s.buff['V']=V
		ni=0

	def test_one_round(self,part_sites,data_index,num_data,num_dim,p1,p2,step_size,max_iter,tol):
		V=part_sites[0].buff['V']
		num_sites=len(part_sites)
		for s in part_sites:
			s.opt(edm_diffusion_gradient_descent_partial,['local_edm','local_entrence','V'],['Vk_tide',step_size,num_sites])
		self.SSP.secure_sum(part_sites,'Vk_tide',part_sites,'V')
		change=abs(V-part_sites[0].buff['V']).sum()
		print(change)
		#V=part_sites[0].buff['V']

'''Local Operation Functions'''
def kappa(A):
	diag=np.repeat(A.diagonal(),A.shape[0]).reshape(A.shape)
	#print(diag)
	#print(A)
	return diag+diag.T-2*A

def kappa_star(A):
	diag=np.diag(A.sum(axis=1))
	#diag=np.repeat(A.diagonal(),A.shape[0]).reshape(A.shape)
	return 2*(diag-A)

def edm_final_completion_partial(site,loc_param_indices,params):
	V=site.buff[loc_param_indices[0]]
	global_edm_index=params[0]
	site.buff[global_edm_index]=kappa(np.dot(V,V.T))

def edm_diffusion_gradient_descent_partial(site,loc_param_indices,params):
	#global FLAG
	local_edm,local_entrence,V=(site.buff[i] for i in loc_param_indices)
	Vk_tide_index,step_size,num_sites=params
	num_data=local_edm.shape[0]
	entrence_matrix=np.zeros((num_data,num_data))
	entrence_matrix[local_entrence[0],local_entrence[1]]=kappa(np.dot(V,V.T))[local_entrence[0],local_entrence[1]]-local_edm[local_entrence[0],local_entrence[1]]

	site.buff[Vk_tide_index]=(V-step_size*np.dot(kappa_star(entrence_matrix),V))/num_sites

def edm_absorb_edm_entrences_partial(site,loc_param_indices,params):
	shared_entrence,local_edm,local_entrence=(site.buff[i] for i in loc_param_indices)
	shared_index,shared_value=shared_entrence
	local_edm[shared_index]=shared_value
	local_entrence[0]=np.hstack((local_entrence[0],shared_index[0]))
	local_entrence[1]=np.hstack((local_entrence[1],shared_index[1]))
	local_edm[shared_index[1],shared_index[0]]=shared_value
	local_entrence[0]=np.hstack((local_entrence[0],shared_index[1]))
	local_entrence[1]=np.hstack((local_entrence[1],shared_index[0]))


def edm_share_data_partial(site,loc_param_indices,params):
	data,array_num_data=(site.buff[i] for i in loc_param_indices)
	shared_index_index,shared_array_index,prop_share,to_share_index,self_index,dcprotocol=params
	num_local_data=data.shape[0]
	num_share=int(num_local_data*prop_share) if prop_share<1 else prop_share
	selected_index=np.random.permutation(range(num_local_data))[:num_share]
	index_offset=array_num_data[:self_index].sum()
	site.buff[shared_index_index].append(selected_index+index_offset)
	site.buff[shared_array_index].append(dcprotocol.encode(data[selected_index]))
	site.buff[to_share_index]=(index_offset+selected_index,data[selected_index])

def edm_shared_edm_entrences_partial(site,loc_param_indices,params):
	data,array_num_data,shared_indices,shared_arraies=(site.buff[i] for i in loc_param_indices)
	local_edm_index,self_index,prop_share,shared_edm_index,edm_computation_func,local_entrence_index,change_index,tol,array_num_data_index=params
	index_offset=array_num_data[:self_index].sum()
	num_data=array_num_data.sum()
	sharing_index=shared_indices[self_index]
	reserved_relative_index=np.delete(np.arange(data.shape[0]),sharing_index - index_offset,axis=0)
	num_reserved=reserved_relative_index.shape[0]
	num_dim=data.shape[1]
	shared_array=np.vstack(shared_arraies)
	shared_index=np.hstack(shared_indices)
	num_shared=shared_index.shape[0]
	local_edm=np.zeros((num_data,num_data))
	updating_index=np.hstack((reserved_relative_index+index_offset,shared_index))
	num_updating=updating_index.shape[0]
	triu_indices=list(np.triu_indices(num_updating,1))
	no_diag_indices=[np.hstack((triu_indices[0],triu_indices[1])),np.hstack((triu_indices[1],triu_indices[0]))]
	local_entrence=[updating_index[no_diag_indices[0]],updating_index[no_diag_indices[1]]]
	local_edm[local_entrence[0],local_entrence[1]]=edm_computation_func(data[reserved_relative_index],shared_array)[no_diag_indices[0],no_diag_indices[1]]
	#local_entrence=[np.repeat(updating_index,updating_index.shape[0]),np.hstack((updating_index for i in range(updating_index.shape[0])))]
	#local_edm[local_entrence[0],local_entrence[1]]=edm_computation_func(data[reserved_relative_index],shared_array).reshape(updating_index.shape[0]**2)
	
	triu_indices[0]=triu_indices[0][:-int((num_shared)*(num_shared -1)/2)]
	triu_indices[1]=triu_indices[1][:-int((num_shared)*(num_shared -1)/2)]
	entrence_to_share_row=np.array([reserved_relative_index[i] for i in triu_indices[0]],dtype=np.int32)+index_offset
	entrence_to_share_column=np.array([updating_index[i] for i in triu_indices[1]],dtype=np.int32)
	num_entrence_to_share=entrence_to_share_row.shape[0]
	num_to_share=int(num_entrence_to_share*prop_share) if prop_share<=1 else prop_share
	sharing_edm_entrence_index=np.random.permutation(np.arange(num_entrence_to_share))[:num_to_share]
	sharing_edm_index=(entrence_to_share_row[sharing_edm_entrence_index],entrence_to_share_column[sharing_edm_entrence_index])
	site.buff={}
	site.buff[local_edm_index]=local_edm
	local_entrence[0]=np.hstack((local_entrence[0],np.arange(num_data)))
	local_entrence[1]=np.hstack((local_entrence[1],np.arange(num_data)))
	site.buff[local_entrence_index]=local_entrence
	#print(sharing_edm_index[0].dtype)
	site.buff[shared_edm_index]=(sharing_edm_index,site.buff[local_edm_index][sharing_edm_index])
	site.buff[change_index]=tol+2
	site.buff[array_num_data_index]=array_num_data

def edm_new_shared_edm_entrences_partial(site,loc_param_indices,params):
	data,array_num_data,shared_indices,shared_arraies=(site.buff[i] for i in loc_param_indices)
	local_edm_index,self_index,prop_share,shared_edm_index,dcprotocol,local_entrence_index,change_index,tol,array_num_data_index=params
	index_offset=array_num_data[:self_index].sum()
	num_data=array_num_data.sum()
	sharing_index=shared_indices[self_index]
	reserved_relative_index=np.delete(np.arange(data.shape[0]),sharing_index - index_offset,axis=0)
	num_reserved=reserved_relative_index.shape[0]
	num_dim=data.shape[1]
	shared_array=np.vstack(shared_arraies)
	shared_index=np.hstack(shared_indices)
	num_shared=shared_index.shape[0]
	local_edm=np.zeros((num_data,num_data))
	updating_index=np.hstack((reserved_relative_index+index_offset,shared_index))
	num_updating=updating_index.shape[0]
	triu_indices=list(np.triu_indices(num_updating,1))
	no_diag_indices=[np.hstack((triu_indices[0],triu_indices[1])),np.hstack((triu_indices[1],triu_indices[0]))]
	
	'''
	specially for distance computation
	'''
	shared_index_noself=np.hstack(shared_indices[:self_index]+shared_indices[self_index+1:])
	shared_array_noself=np.vstack(shared_arraies[:self_index]+shared_arraies[self_index+1:])
	updating_index_noself=np.hstack((np.arange(data.shape[0])+index_offset,shared_index_noself))
	local_entrence=[updating_index_noself[no_diag_indices[0]],updating_index_noself[no_diag_indices[1]]]
	local_edm[local_entrence[0],local_entrence[1]]=dcprotocol.distances(data,shared_array_noself)[no_diag_indices[0],no_diag_indices[1]]
	#local_entrence=[np.repeat(updating_index,updating_index.shape[0]),np.hstack((updating_index for i in range(updating_index.shape[0])))]
	#local_edm[local_entrence[0],local_entrence[1]]=edm_computation_func(data[reserved_relative_index],shared_array).reshape(updating_index.shape[0]**2)
	
	triu_indices[0]=triu_indices[0][:-int((num_shared)*(num_shared -1)/2)]
	triu_indices[1]=triu_indices[1][:-int((num_shared)*(num_shared -1)/2)]
	entrence_to_share_row=np.array([reserved_relative_index[i] for i in triu_indices[0]],dtype=np.int32)+index_offset
	entrence_to_share_column=np.array([updating_index[i] for i in triu_indices[1]],dtype=np.int32)
	num_entrence_to_share=entrence_to_share_row.shape[0]
	num_to_share=int(num_entrence_to_share*prop_share) if prop_share<=1 else prop_share
	sharing_edm_entrence_index=np.random.permutation(np.arange(num_entrence_to_share))[:num_to_share]
	sharing_edm_index=(entrence_to_share_row[sharing_edm_entrence_index],entrence_to_share_column[sharing_edm_entrence_index])
	site.buff={}
	site.buff[local_edm_index]=local_edm
	local_entrence[0]=np.hstack((local_entrence[0],np.arange(num_data)))
	local_entrence[1]=np.hstack((local_entrence[1],np.arange(num_data)))
	site.buff[local_entrence_index]=local_entrence
	#print(sharing_edm_index[0].dtype)
	site.buff[shared_edm_index]=(sharing_edm_index,site.buff[local_edm_index][sharing_edm_index])
	site.buff[change_index]=tol+2
	site.buff[array_num_data_index]=array_num_data


def edm_count_data_partial(site,loc_param_indices,params):
	data=site.buff[loc_param_indices[0]]
	local_num_data_index,array_num_data_index,global_num_data_index,site_id,num_sites=params
	site.buff[local_num_data_index]=[site_id,data.shape[0]]
	site.buff[array_num_data_index]=np.zeros(num_sites,dtype=np.int32)
	site.buff[array_num_data_index][site_id]=data.shape[0]
	site.buff[global_num_data_index]=data.shape[0]

def edm_save_local_num_data_partial(site,loc_param_indices,params):
	array_num_data,[received_id,received_num_data]=(site.buff[i] for i in loc_param_indices)
	global_num_data_index=params[0]
	array_num_data[received_id]=received_num_data
	site.buff[global_num_data_index]+=received_num_data

'''Random Projection-based Multiplicative Data Perturbation'''
class MultiPerturbPPSC(DistributedProtocol):
	def __init__(self,sigma=1):
		DistributedProtocol.__init__(self)
		self.sigma=1

	def setRdirect(self,R):
		self.R=R

	def genR(self,m,d):
		self.R=np.random.normal(0,1,(d,m))

	def encode(self,data):
		m=self.R.shape[1]
		return np.dot(data,self.R)/(np.sqrt(m)*self.sigma)

	def distances(self,reserved_data,shared_data):
		if shared_data.shape[0]==0:
			return euclidean_distances(reserved_data,squared=True)
		num_reserved=reserved_data.shape[0]
		num_shared=shared_data.shape[0]
		num_all=num_reserved+num_shared
		m=self.R.shape[1]
		perturbed_reserved_data=self.encode(reserved_data)
		Y_XY=euclidean_distances(shared_data,np.vstack((perturbed_reserved_data,shared_data)),squared=True)
		dis=np.zeros((num_all,num_all))
		dis[:num_reserved,:num_reserved]=euclidean_distances(reserved_data,squared=True)
		dis[num_reserved:,:]=Y_XY
		dis[:num_reserved:,num_reserved:]=Y_XY[:,:num_reserved].T
		return dis


'''
Distributed LapRLS
'''
class DMR(DistributedProtocol):
	def __init__(self,SecureSummationProtocol,EDMcompletionProtocol):
		DistributedProtocol.__init__(self)
		self.SSP=SecureSummationProtocol
		self.EDMcom=EDMcompletionProtocol

	def fit_with_edm(self,part_sites,data_index,label_index,array_num_data_index,gammaA,gammaI,sigma,nn,num_class):
		num_sites=len(part_sites)
		for i in range(num_sites):
			part_sites[i].opt(dmr_laplacian_estimation_partial,['global_edm'],['K','Laplacian',sigma,nn])
			part_sites[i].opt(dmr_preparing_labeled_entrence_partial,['array_num_data',label_index],['Jk','J',i,num_sites])
		self.SSP.secure_sum(part_sites,'J',part_sites,'J')
		for i in range(num_sites):
			part_sites[i].opt(dmr_local_training_partial,['J','Jk','K','Laplacian',label_index],['alpha',gammaA,gammaI,num_sites,num_class])
		self.SSP.secure_sum(part_sites,'alpha',part_sites,'alpha')
		for i in range(num_sites):
			part_sites[i].opt(dmr_extract_local_weight,['alpha','array_num_data'],[i,'alpha'])

	def label(self,part_sites,new_data,sigma):
		num_sites=len(part_sites)
		for s in part_sites:
			s.opt(dmr_labeling_partial,['alpha','data'],[sigma,new_data,'new_label',num_sites])
		self.SSP.secure_sum(part_sites,'new_label',part_sites,'new_label')
		label_matrix=part_sites[0].buff['new_label']
		label_matrix=label_matrix/label_matrix.sum(axis=0)
		label_matrix=label_matrix.argmax(axis=1).reshape((label_matrix.shape[0],1))+1
		for s in part_sites:
			s.buff['new_label']=label_matrix
		return label_matrix

def load_data(part_sites,data_index,data_list):
	for i in range(len(part_sites)):
		part_sites[i].buff[data_index]=data_list[i]

def dmr_labeling_partial(site,loc_param_indices,params):
	alpha,local_data=(site.buff[i] for i in loc_param_indices)
	sigma,new_data,local_new_label_index,num_sites=params
	distances=euclidean_distances(new_data,local_data,squared=True)
	kernels=np.exp(-distances/(2*sigma))
	site.buff[local_new_label_index]=np.dot(kernels,alpha)#/num_sites

def dmr_extract_local_weight(site,loc_param_indices,params):
	alpha,array_num_data=(site.buff[i] for i in loc_param_indices)
	self_index,alpha_index=params
	local_index=array_num_data[:self_index].sum()+np.arange(array_num_data[self_index])
	site.buff[alpha_index]=alpha[local_index]

def dmr_local_training_partial(site,loc_param_indices,params):
	J,Jk,K,Laplacian,labels=(site.buff[i] for i in loc_param_indices)
	alpha_index,gammaA,gammaI,num_sites,num_class=params
	temp=np.dot((J+gammaI*Laplacian),K)
	label_matrix=np.zeros((labels.shape[0],num_class))
	labeled_index=np.where(labels!=0)[0]
	#label_matrix[labeled_index]=-1
	label_matrix[labeled_index,labels[labeled_index].astype(np.int32).reshape(labeled_index.shape[0])-1]=1
	site.buff['label_matrix']=label_matrix
	#temp=np.dot(J,K)+np.dot(K,gammaI*Laplacian)
	temp+=gammaA*np.diag(np.ones(temp.shape[0]))
	site.buff[alpha_index]=np.dot(np.linalg.inv(temp),np.dot(Jk.T,label_matrix))

def dmr_laplacian_estimation_partial(site,loc_param_indices,params):
	edm=site.buff[loc_param_indices[0]]
	kernel_index,Laplacian_index,sigma,nn=params
	kernels=np.exp(-edm/(2*sigma))
	site.buff[kernel_index]=kernels
	weights=np.zeros(edm.shape)
	largests_indices=np.hstack(edm.argsort(axis=1)[:,:nn])
	weights[np.repeat(np.arange(edm.shape[0]),nn),largests_indices]=1
	sqrt_row_sum=np.sqrt(weights.sum(axis=1))
	site.buff[Laplacian_index]=(weights/sqrt_row_sum)/sqrt_row_sum.reshape((sqrt_row_sum.shape[0],1))

def dmr_preparing_labeled_entrence_partial(site,loc_param_indices,params):
	array_num_data,labels=(site.buff[i] for i in loc_param_indices)
	local_J_index,global_J_index,self_index,num_sites=params
	index_offset=array_num_data[:self_index].sum()
	num_data=array_num_data.sum()
	local_num_data=array_num_data[self_index]
	Jk=np.zeros((local_num_data,num_data))
	num_class=labels.shape[1]
	labeled_index=np.arange(local_num_data)[(labels!=0).all(axis=1)]
	Jk[labeled_index,labeled_index+index_offset]=1
	site.buff[local_J_index]=Jk
	site.buff[global_J_index]=np.dot(Jk.T,Jk)

