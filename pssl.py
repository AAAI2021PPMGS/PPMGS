'''
An implementation of the semi-supervised DPPDM protocol, PSSL

citation:
@InProceedings{Gueler2019,
  author    = {Basak G{\"{u}}ler and Amir Salman Avestimehr and Antonio Ortega},
  title     = {Privacy-Aware Distributed Graph-Based Semi-Supervised Learning},
  booktitle = {29th {IEEE} International Workshop on Machine Learning for Signal Processing, {MLSP} 2019, Pittsburgh, PA, USA, October 13-16, 2019},
  year      = {2019},
  pages     = {1--6},
  publisher = {{IEEE}},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl    = {https://dblp.org/rec/conf/mlsp/GulerAO19.bib},
  doi       = {10.1109/MLSP.2019.8918797},
}
'''

import numpy as np
import time

from sklearn.cluster import KMeans as KM
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors as NN
from scipy import sparse as sp

from simulator import DistributedProtocol,TINY_POSITIVE

'''
Sub-Protocol: Secure 2-party Computation
'''
'''Protocol Class'''
class Secure2PC(DistributedProtocol):
	def __init__(self,fake_performance=False):
		DistributedProtocol.__init__(self)
		self.fake_performance=fake_performance
	
	def share(self,site0,site1,term_index,local_share_index,foreign_share_index):
		site0.opt(s2pc_share_partial,[term_index],[local_share_index,'s2pc_temp'])
		site0.send([site1],[foreign_share_index],['s2pc_temp'])

	def sum(self,site0,site1,term0_index0,term1_index0,term0_index1,term1_index1,rec_index):
		site0.opt(s2pc_sum_partial,[term0_index0,term1_index0],[rec_index])
		site1.opt(s2pc_sum_partial,[term0_index1,term1_index1],[rec_index])
		
	def mult(self,site0,site1,factor0_index0,factor1_index0,factor0_index1,factor1_index1,rec_index,triplet0=[1,1,2],triplet1=[1,1,2]):
		site0.opt(s2pc_mult_partial,[factor0_index0,factor1_index0],['s2pc_mult_e','s2pc_mult_f',triplet0[0],triplet0[1]])
		site1.opt(s2pc_mult_partial,[factor0_index1,factor1_index1],['s2pc_mult_e','s2pc_mult_f',triplet1[0],triplet1[1]])
		self.recover(site0,site1,'s2pc_mult_e','s2pc_mult_e','s2pc_mult_e')
		site0.send([site1],['s2pc_mult_e'],['s2pc_mult_e'])
		self.recover(site0,site1,'s2pc_mult_f','s2pc_mult_f','s2pc_mult_f')
		site0.send([site1],['s2pc_mult_f'],['s2pc_mult_f'])
		site0.opt(s2pc_mult2_partial,['s2pc_mult_e','s2pc_mult_f',factor0_index0,factor1_index0],[0,triplet0[2],rec_index])
		site1.opt(s2pc_mult2_partial,['s2pc_mult_e','s2pc_mult_f',factor0_index1,factor1_index1],[1,triplet1[2],rec_index])

	def recover(self,site0,site1,local_share_index,foreign_share_index,rec_index):
		site1.send([site0],['s2pc_temp'],[foreign_share_index])
		site0.opt(s2pc_recover_partial,[local_share_index,'s2pc_temp'],[rec_index])

	def metric(self,site0,site1,value0_index,value1_index,value0_share_index0,value1_share_index0,value0_share_index1,value1_share_index1,rec_index,triplet0=[1,1,2],triplet1=[1,1,2]):
		self.mult(site0,site1,value0_share_index0,value1_share_index0,value0_share_index1,value1_share_index1,'s2pc_temp',triplet0,triplet1)
		site0.opt(s2pc_metric_partial,[value0_index,'s2pc_temp'],[rec_index])
		site1.opt(s2pc_metric_partial,[value1_index,'s2pc_temp'],[rec_index])
		if self.fake_performance:
			result=(site0.buff[value0_index]-site1.buff[value1_index])**2
			site0.buff[rec_index]=result/2
			site1.buff[rec_index]=result/2

	def compare_public(self,site0,site1,share_index0,share_index1,competitor,rec_index,triplet0=[1,1,2],triplet1=[1,1,2]):
		site0.opt(s2pc_less_public_partial,[share_index0],[competitor,'s2pc_random',rec_index])
		site1.opt(s2pc_less_public_partial,[share_index1],[competitor,'s2pc_random',rec_index])
		self.mult(site0,site1,rec_index,'s2pc_random',rec_index,'s2pc_random',rec_index,triplet0,triplet1)
		if self.fake_performance:
			result=site0.buff[share_index0]+site1.buff[share_index1]-competitor
			site0.buff[rec_index]=result/2
			site1.buff[rec_index]=result/2

	def compare_private(self,site0,site1,share_index0,share_index1,competitor_index0,competitor_index1,rec_index,triplet0=[1,1,2],triplet1=[1,1,2]):
		#print(competitor_index0)
		site0.opt(s2pc_less_private_partial,[share_index0,competitor_index0],['s2pc_random',rec_index])
		site1.opt(s2pc_less_private_partial,[share_index1,competitor_index1],['s2pc_random',rec_index])
		self.mult(site0,site1,rec_index,'s2pc_random',rec_index,'s2pc_random',rec_index,triplet0,triplet1)
		if self.fake_performance:
			result=site0.buff[share_index0]+site1.buff[share_index1]-site0.buff[competitor_index0]-site1.buff[competitor_index1]
			site0.buff[rec_index]=result/2
			site1.buff[rec_index]=result/2

'''Local Operation Functions'''
def s2pc_share_partial(site,loc_param_indices,params):
	term=site.buff[loc_param_indices[0]]
	self_share_index,to_share_index=params
	self_share=np.random.random()*term
	#print(self_share,term)
	site.buff[self_share_index]=self_share
	site.buff[to_share_index]=term - self_share

def s2pc_sum_partial(site,loc_param_indices,params):
	term0,term1=(site.buff[i] for i in loc_param_indices)
	rec_index=params[0]
	site.buff[rec_index]=term0+term1

def s2pc_recover_partial(site,loc_param_indices,params):
	local_share,foreign_share=(site.buff[i] for i in loc_param_indices)
	rec_index=params[0]
	site.buff[rec_index]=local_share+foreign_share

def s2pc_mult_partial(site,loc_param_indices,params):
	factor0,factor1=(site.buff[i] for i in loc_param_indices)
	local_e_index,local_f_index,local_u,local_v=params
	site.buff[local_e_index]=factor0-local_u
	site.buff[local_f_index]=factor1-local_v

def s2pc_mult2_partial(site,loc_param_indices,params):
	global_e,global_f,factor0,factor1=(site.buff[i] for i in loc_param_indices)
	footnote_i,local_z,rec_index=params
	site.buff[rec_index]=-footnote_i*global_e*global_f+global_f*factor0+global_e*factor1+local_z

def s2pc_metric_partial(site,loc_param_indices,params):
	value_share,product_share=(site.buff[i] for i in loc_param_indices)
	rec_index=params[0]
	res=value_share**2-2*product_share
	if type(res)==np.ndarray:
		res=res.sum()
	site.buff[rec_index]=res

def s2pc_less_public_partial(site,loc_param_indices,params):
	value_share=site.buff[loc_param_indices[0]]
	competitor,random_index,rec_index=params
	site.buff[rec_index]=value_share-competitor/2
	site.buff[random_index]=np.random.random()*2

def s2pc_less_private_partial(site,loc_param_indices,params):
	#print(loc_param_indices)
	value_share,competitor_share=(site.buff[i] for i in loc_param_indices)
	random_index,rec_index=params
	site.buff[rec_index]=value_share-competitor_share
	site.buff[random_index]=np.random.random()*2

'''
PSSL
'''
'''Protocol Class'''
class PASSL(DistributedProtocol):
	def __init__(self,Secure2PCProtocol,SecureSummationProtocol):
		DistributedProtocol.__init__(self)
		self.SSP=SecureSummationProtocol
		self.S2PCP=Secure2PCProtocol

	def graph_construction(self,part_sites,data_index,K,R,dmin,dmax,lamb,rbf_sigma,num_class):
		for site in part_sites:
			site.opt(passl_local_graph_partial,[data_index],[K,rbf_sigma,'passl_local_graph',num_class,'passl_local_center','passl_local_cluster','passl_inter_graphs','passl_member_id'])
		num_sites=len(part_sites)
		for i in range(num_sites):
			for j in range(i+1,num_sites):
				part_sites[i].opt(passl_inter_site_init_partial,['passl_member_id'],['passl_count_point_connection'])
				part_sites[j].opt(passl_inter_site_init_partial,['passl_member_id'],['passl_count_point_connection'])
				self.inter_graph_2P(part_sites[i],part_sites[j],data_index,K,R,dmin,dmax,lamb,rbf_sigma,num_class,i,j,'passl_inter_graphs')
				#self.inter_graph_2P(part_sites[j],part_sites[i],data_index,K,R,dmin,dmax,lamb,rbf_sigma,num_class,i,'passl_inter_graphs')
					
	def distributed_label_propagation(self,part_sites,alpha,label_index,num_iter,num_class):
		#passl_distributedLP_initial_partial()
		for s in part_sites:
			s.opt(passl_distributedLP_initial_partial,[label_index,'passl_local_graph','passl_inter_graphs'],[alpha,'passl_D','passl_A','passl_Y',num_class])
		num_sites=len(part_sites)
		for h in range(num_iter):
			for i in range(num_sites):
				part_sites[i].opt(passl_distributedLP_sponsor_partial,['passl_A','passl_local_graph'],['s2pc_term'])
				for j in range(num_sites):
					if not j==i:
						part_sites[j].opt(passl_distributedLP_participant_partial,['passl_A','passl_inter_graphs'],['s2pc_term',i])
				self.SSP.secure_sum(part_sites,'s2pc_term',[part_sites[i]],'s2pc_res')
				part_sites[i].opt(passl_distributedLP_propagation_partial,['passl_D','passl_Y','s2pc_res'],['passl_A'])

	def label(self,part_sites,new_data,data_index,sigma):
		num_sites=len(part_sites)
		for s in part_sites:
			s.opt(passl_labeling_partial,['passl_A',data_index],[sigma,new_data,'new_label',num_sites])
		self.SSP.secure_sum(part_sites,'new_label',part_sites,'new_label')
		label_matrix=part_sites[0].buff['new_label']
		label_matrix=label_matrix/label_matrix.sum(axis=0)
		#return label_matrix
		label_matrix=label_matrix.argmax(axis=1).reshape((label_matrix.shape[0],1))+1
		for s in part_sites:
			s.buff['new_label']=label_matrix
		return label_matrix.flatten()

	def inter_graph_2P(self,sitei,sitej,data_index,K,R,dmin,dmax,lamb,rbf_sigma,num_class,iindex,jindex,inter_graph_index):
		indices_i=[]
		indices_j=[]
		distances=[]
		for li in range(num_class):
			sitei.buff['s2pc_term']=sitei.buff['passl_local_center'][li]
			self.S2PCP.share(sitei,sitej,'s2pc_term','s2pc_local_share','s2pc_foreign_share')
			sitei.buff['passl_min_center_dist']=1e10
			sitej.buff['passl_min_center_dist']=1e10
			count_connection=0
			for lj in range(num_class):
				sitej.buff['s2pc_term']=sitej.buff['passl_local_center'][lj]
				self.S2PCP.share(sitej,sitei,'s2pc_term','s2pc_local_share','s2pc_foreign_share')
				self.S2PCP.metric(sitei,sitej,'s2pc_term','s2pc_term','s2pc_local_share','s2pc_foreign_share','s2pc_foreign_share','s2pc_local_share','s2pc_res')
				self.S2PCP.compare_private(sitei,sitej,'s2pc_res','s2pc_res','passl_min_center_dist','passl_min_center_dist','s2pc_res2')
				self.S2PCP.recover(sitei,sitej,'s2pc_res2','s2pc_res2','s2pc_res2')
				#print(sitei.buff['s2pc_res']+sitej.buff['s2pc_res'])
				#print(sitei.buff['s2pc_res2'])
				if sitei.buff['s2pc_res2']<0:
					#print(li,lj)
					#print(sitei.buff['s2pc_res']+sitej.buff['s2pc_res'])
					sitei.buff['passl_min_center_dist']=sitei.buff['s2pc_res']
					sitej.buff['passl_min_center_dist']=sitej.buff['s2pc_res']
					min_center=lj
			#print(min_center)
			count_point_connection_i=sitei.buff['passl_count_point_connection'][li]
			count_point_connection_j=sitej.buff['passl_count_point_connection'][min_center]
			current_cluster_i=sitei.buff['passl_member_id'][li]
			current_cluster_j=sitej.buff['passl_member_id'][min_center]
			#print(current_cluster_j)
			num_xi=current_cluster_i.shape[0]
			num_xj=current_cluster_j.shape[0]

			'''
			test
			'''
			candidates={i:list(range(num_xj)) for i in range(num_xi)}
			for shuffle_id in range(num_xi):
				np.random.shuffle(candidates[shuffle_id])
			while len(candidates)>0:
				if count_connection>=R:
					break
				xi=np.random.choice(tuple(candidates.keys()))
				#print(xi,candidates[xi])
				abs_xi=current_cluster_i[xi]
				xj=candidates[xi].pop()
				if count_point_connection_j[xj]<lamb:
					sitei.buff['s2pc_term']=sitei.buff[data_index][abs_xi]
					self.S2PCP.share(sitei,sitej,'s2pc_term','s2pc_local_share','s2pc_foreign_share')
					abs_xj=current_cluster_j[xj]
					sitej.buff['s2pc_term']=sitej.buff[data_index][abs_xj]
					self.S2PCP.share(sitej,sitei,'s2pc_term','s2pc_local_share','s2pc_foreign_share')
					self.S2PCP.metric(sitei,sitej,'s2pc_term','s2pc_term','s2pc_local_share','s2pc_foreign_share','s2pc_foreign_share','s2pc_local_share','s2pc_res')
					#print(sitei.buff['s2pc_res']+sitej.buff['s2pc_res'],abs_xi,abs_xj)
					self.S2PCP.compare_public(sitei,sitej,'s2pc_res','s2pc_res',dmin,'s2pc_res2')
					self.S2PCP.recover(sitei,sitej,'s2pc_res2','s2pc_res2','s2pc_res2')
					if sitei.buff['s2pc_res2']>0:
						self.S2PCP.compare_public(sitei,sitej,'s2pc_res','s2pc_res',dmax,'s2pc_res2')
						self.S2PCP.recover(sitei,sitej,'s2pc_res2','s2pc_res2','s2pc_res2')
						if sitei.buff['s2pc_res2']<0:
							#print(abs_xi,abs_xj,li,min_center)
							indices_i.append(abs_xi)
							indices_j.append(abs_xj)
							self.S2PCP.recover(sitei,sitej,'s2pc_res','s2pc_res','s2pc_res')
							#self.S2PCP.recover(part_sites[j],part_sites[j],'s2pc_res','s2pc_res')
							distances.append(sitei.buff['s2pc_res'])
							count_connection+=1
							count_point_connection_i[xi]+=1
							count_point_connection_j[xj]+=1
				if len(candidates[xi])<1:
					_=candidates.pop(xi)

			'''
			end
			'''
			'''
			for xi in range(num_xi):
				if count_connection>=R:
					break
				abs_xi=current_cluster_i[xi]
				sitei.buff['s2pc_term']=sitei.buff[data_index][abs_xi]
				self.S2PCP.share(sitei,sitej,'s2pc_term','s2pc_local_share','s2pc_foreign_share')
				for xj in range(num_xj):
					if count_point_connection_i[xi]>=lamb or count_connection>=R:
						break
					if count_point_connection_j[xj]<lamb:
						abs_xj=current_cluster_j[xj]
						sitej.buff['s2pc_term']=sitej.buff[data_index][abs_xj]
						self.S2PCP.share(sitej,sitei,'s2pc_term','s2pc_local_share','s2pc_foreign_share')
						self.S2PCP.metric(sitei,sitej,'s2pc_term','s2pc_term','s2pc_local_share','s2pc_foreign_share','s2pc_foreign_share','s2pc_local_share','s2pc_res')
						#print(sitei.buff['s2pc_res']+sitej.buff['s2pc_res'],abs_xi,abs_xj)
						self.S2PCP.compare_public(sitei,sitej,'s2pc_res','s2pc_res',dmin,'s2pc_res2')
						self.S2PCP.recover(sitei,sitej,'s2pc_res2','s2pc_res2','s2pc_res2')
						if sitei.buff['s2pc_res2']>0:
							self.S2PCP.compare_public(sitei,sitej,'s2pc_res','s2pc_res',dmax,'s2pc_res2')
							self.S2PCP.recover(sitei,sitej,'s2pc_res2','s2pc_res2','s2pc_res2')
							if sitei.buff['s2pc_res2']<0:
								#print(abs_xi,abs_xj,li,min_center)
								indices_i.append(abs_xi)
								indices_j.append(abs_xj)
								self.S2PCP.recover(sitei,sitej,'s2pc_res','s2pc_res','s2pc_res')
								#self.S2PCP.recover(part_sites[j],part_sites[j],'s2pc_res','s2pc_res')
								distances.append(sitei.buff['s2pc_res'])
								count_connection+=1
								count_point_connection_i[xi]+=1
								count_point_connection_j[xj]+=1
			'''
		sitei.opt(passl_inter_graph_partial,None,[indices_i,indices_j,distances,jindex,inter_graph_index,sitei.buff[data_index].shape[0],sitej.buff[data_index].shape[0],rbf_sigma])
		sitej.opt(passl_inter_graph_partial,None,[indices_j,indices_i,distances,iindex,inter_graph_index,sitej.buff[data_index].shape[0],sitei.buff[data_index].shape[0],rbf_sigma])

'''Local Operation Functions'''
def passl_local_graph_partial(site,loc_param_indices,params):
	X=site.buff[loc_param_indices[0]]
	K,rbf_sigma,local_graph_index,n_cluster,centers_index,point_cluster_index,inter_graph_index,member_id_index=params
	nins=NN(K+1,None,metric='euclidean').fit(X)
	W=nins.kneighbors_graph(nins._fit_X,K+1,mode='distance')
	#W.data=W.data**2
	W.data=np.exp(-W.data**2/rbf_sigma)
	W[np.diag_indices(W.shape[0])]=0
	#W[np.diag_indices(W.shape[0])]=0
	site.buff[local_graph_index]=W
	kins=KM(n_cluster)
	point_cluster=kins.fit_predict(X)
	site.buff[point_cluster_index]=point_cluster
	site.buff[centers_index]=kins.cluster_centers_
	#print(kins.cluster_centers_)
	site.buff[inter_graph_index]={}
	member_id=[]
	for i in range(n_cluster):
		member_id.append(np.where(point_cluster==i)[0])
		#print(member_id[-1])
	site.buff[member_id_index]=member_id

def passl_inter_site_init_partial(site,loc_param_indices,params):
	point_cluster=site.buff[loc_param_indices[0]]
	count_point_connection_index=params[0]
	site.buff[count_point_connection_index]=[np.zeros(i.shape[0]) for i in point_cluster]

def passl_inter_graph_partial(site,loc_param_indices,params):
	indices_i,indices_j,distances,jindex,inter_graph_index,num_data_i,num_data_j,rbf_sigma=params
	inter_graph=sp.csr_matrix((distances,(indices_j,indices_i)),(num_data_j,num_data_i))
	inter_graph.data=np.exp(-inter_graph.data/rbf_sigma)
	site.buff[inter_graph_index][jindex]=inter_graph

def passl_distributedLP_initial_partial(site,loc_param_indices,params):
	labels,local_graph,inter_graphs=(site.buff[i] for i in loc_param_indices)
	alpha,D_index,A_index,Y_index,num_class=params
	label_matrix=np.zeros((labels.shape[0],num_class),dtype=np.int32)
	labeled_index=np.where(labels!=0)[0]
	#label_matrix[labeled_index]=-1
	label_matrix[labeled_index,labels[labeled_index].astype(np.int32).reshape(labeled_index.shape[0])-1]=1
	site.buff[A_index]=label_matrix
	D=local_graph.sum(axis=0).getA().flatten()
	for g in inter_graphs.values():
		D+=g.sum(axis=0).getA().flatten()
	site.buff[D_index]=alpha/(D+TINY_POSITIVE)
	site.buff[Y_index]=(1-alpha)*label_matrix

def passl_distributedLP_sponsor_partial(site,loc_param_indices,params):
	A,local_graph=(site.buff[i] for i in loc_param_indices)
	term_index=params[0]
	site.buff[term_index]=local_graph.dot(A)

def passl_distributedLP_participant_partial(site,loc_param_indices,params):
	A,inter_graphs=(site.buff[i] for i in loc_param_indices)
	term_index,sponsor_index=params
	#print(inter_graphs[sponsor_index].shape,A.shape)
	site.buff[term_index]=inter_graphs[sponsor_index].dot(A)

def passl_distributedLP_propagation_partial(site,loc_param_indices,params):
	D,Y,W=(site.buff[i] for i in loc_param_indices)
	A_index=params[0]
	site.buff[A_index]=D[:,np.newaxis]*W+Y

def passl_labeling_partial(site,loc_param_indices,params):
	A,local_data=(site.buff[i] for i in loc_param_indices)
	sigma,new_data,local_new_label_index,num_sites=params
	distances=euclidean_distances(new_data,local_data,squared=True)
	kernels=np.exp(-distances/sigma)
	site.buff[local_new_label_index]=np.dot(kernels,A)