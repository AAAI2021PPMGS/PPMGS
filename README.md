##Privacy-preserving Mixture-distribution-based Graph Smoothing

	This is a local simulation of the semi-supervised distributed privacy-preserving data mining (DPPDM) protocol, PPMGS.

	All operations of each participant in the distributed system is simulated locally.	

	Each participant simulator maintains a python dict as its local memory.

	When inter-participant commnication happens, it simply assigns the data of the sender to the designated index in the dict of the receiver and records the size of the data.

	Baseline methods are also implemented, including 
	
	 
	*PSSL
	
	citation:
	
	@InProceedings{Gueler2019,
	
	author		= {Basak G{\"{u}}ler and Amir Salman Avestimehr and Antonio Ortega},
	
	title		 = {Privacy-Aware Distributed Graph-Based Semi-Supervised Learning},
	
	booktitle = {29th {IEEE} International Workshop on Machine Learning for Signal Processing, {MLSP} 2019, Pittsburgh, PA, USA, October 13-16, 2019},
	
	year			= {2019},
	
	pages		 = {1--6},
	
	publisher = {{IEEE}},
	
	bibsource = {dblp computer science bibliography, https://dblp.org},
	
	biburl		= {https://dblp.org/rec/conf/mlsp/GulerAO19.bib},
	
	doi			 = {10.1109/MLSP.2019.8918797},
	
	}
	 

	*DLR
	
	citation:
	
	@Article{Fierimonte2017,
	
	author		= {Roberto Fierimonte and Simone Scardapane and Aurelio Uncini and Massimo Panella},
	
	title		 = {Fully Decentralized Semi-supervised Learning via Privacy-preserving Matrix Completion},
	
	journal	 = {{IEEE} Trans. Neural Networks Learn. Syst.},
	
	year			= {2017},
	
	volume		= {28},
	
	number		= {11},
	
	pages		 = {2699--2711},
	
	bibsource = {dblp computer science bibliography, https://dblp.org},
	
	biburl		= {https://dblp.org/rec/journals/tnn/FierimonteSUP17.bib},
	
	doi			 = {10.1109/TNNLS.2016.2597444},
	
	}


	*HEM
	
	citation:
	
	@InProceedings{Zhu2003,
	
	author				= {Xiaojin Zhu and Zoubin Ghahramani and John D. Lafferty},
	
	title				 = {Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions},
	
	booktitle		 = {Machine Learning, Proceedings of the Twentieth International Conference {(ICML} 2003), August 21-24, 2003, Washington, DC, {USA}},
	
	year					= {2003},
	
	editor				= {Tom Fawcett and Nina Mishra},
	
	pages				 = {912--919},
	
	publisher		 = {{AAAI} Press},
	
	bibsource		 = {dblp computer science bibliography, https://dblp.org},
	
	biburl				= {https://dblp.org/rec/conf/icml/ZhuGL03.bib},
	
	url					 = {http://www.aaai.org/Library/ICML/2003/icml03-118.php},
	
	}


##Requirements

	*sci-kit learn
	
	*numpy
	
	*scipy



##Run the demo

	*PPMGS:
	
	python demo_PPMGS.py


	*PSSL:

	python demo_PSSL.py


	*DLR:
	
	python demo_DLR.py


	*HEM:

	python demo_HEM.py


##Available Parameters:
	
	*PPMGS:
	
	--K=$integer : The number of mixture components; default: 2.
	
	--l=$integer : The number of labeled data per participant; default:7.
	
	--n=$integer : The number of total training data per participant; default:52.
	
	--m=$integer : The number of participants, default 7.
	
	--r=$float : The hyper-parameter denoting the number of the assumed unlabeled random variables, non-integer allowed; default: 1.
	
	--sigma=$float : The hyper-parameter denoting the bandwidth in similarity computation; default: 10.


	*PSSL:
	
	--l=$integer : The number of labeled data per participant; default:7.
	
	--n=$integer : The number of total training data per participant; default:52.
	
	--m=$integer : The number of participants, default 7.
	
	--sigma=$float : The hyper-parameter denoting the bandwidth in similarity computation; default: 17.5**2.
	
	--max_iter=$integer : The number of iterations for label propagation; defalut: 20.
	
	--dmin=$float : The closest distances allowed to connect in inter-participant graphs; default: 0.
	
	--dmax=$float : The furthest distances allowed to connect in inter-participant graphs; default: 150.
	
	--K=$integer : The number of nearest neighors to connect in local graph; default: 7.
	
	--R=$integer : The number of connections built between two clusters; default: 200.
	
	--lamb=$integer : The maximum number of connections per participant for a node; default: 7.
	
	--alpha=$ineger : The label propagation hyper-parameter; default: 0.5.


	*DLR
	
	--l=$integer : The number of labeled data per participant; default:7.
	
	--n=$integer : The number of total training data per participant; default:52.
	
	--m=$integer : The number of participants, default 7.
	
	--sigma=$float : The hyper-parameter denoting the bandwidth in similarity computation; default: 17.5**2.
	
	--p1=$integer : The number of instances to share per participant; default: 10.
	
	--p2=$integer : The number of elements of local EDM to share per participant; default: 890.
	
	--step_size=$float : Step size for EDM completion; default: 1e-5.
	
	--max_iter=$integer : The maximum number of iterations of EDM completion; default: 1500.
	
	--tol=$float : The tolerance for EDM change; default: 1e-3.
	
	--edm_rank=$integer : assumed EDM rank; default: 48.
	
	--gammaA=$float : Regulariztion parameter gammaA; default: 1e-2.
	
	--gammaI=$float : Regulariztion parameter gammaI; default: 1e-2.
	
	--m_rpm=$integer : The number of projecting dimension in multicative perturbation; default: 48.
	
	--nn=$integer : The number of nearest neighors in graph; default: 50.
 

	*HEM
	
	--local : If this argument is given, the demo will run a local version of HEM, otherwise a centralized version; default: centralized.
	
	--l=$integer : The number of labeled data per participant; default:7.
	
	--n=$integer : The number of total training data per participant; default:52.
	
	--m=$integer : The number of participants, default 7.
	
	--sigma=$float : The hyper-parameter denoting the bandwidth in similarity computation; default: 17.5**2.
	
	
	*All demos are run on the G50C dataset.
