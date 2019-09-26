from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.initial_tension import *
from scipy.optimize import fsolve

Activation_Bounds = [[0,1],[0,1]]

def return_random_initial_muscle_lengths_and_activations(InitialTension,X_o,**kwargs):
	"""
	This function returns initial muscle lengths and muscle activations for a given pretensioning level, as derived from (***insert file_name here for scratchwork***) for the system that starts from rest. (Ex. pendulum_eqns.reference_trajectories._01).

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**kwargs

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Seed - Can see the random tension generated. When FixedInitialTension is provided, this seed will apply only to the initial conditions for activation and muscle length.

	2) PlotBool - Must be either True or False. Default is False. Will plot all possible initial muscle lengths and activations for a given pretensioning level.

	3) InitialTensionAcceleration - must be a numpy array of shape (2,). Default is set to the value generated from joint angle IC's for pendulum_eqns.reference_trajectories._01.py (B+A,0,-Aw²,0,Aw⁴,etc.). If using a different reference trajectory, it would be best to either set all joint angle IC's to zero (like for pendulum_eqns.reference_trajectories._02.py) or to derive the value and pass it through the kwargs here (NOTE: must also consider how changing angular IC's effect other state derivative IC's).

	3) InitialAngularAcceleration - must be either a numpy.float64, float, or int. Default is 0. Choice of reference trajectory *should* not matter as it is either 0 or d2r(0) (either by convention or by choice).

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""
	PlotBool = kwargs.get("PlotBool",False)
	assert type(PlotBool)==bool,"PlotBool must be a boolean. Default is False."

	InitialAngularAcceleration = kwargs.get(
		"InitialAngularAcceleration",
		d2r(0)
	) # 0 or d2r(0)
	assert str(type(InitialAngularAcceleration)) in ["<class 'float'>","<class 'int'>","<class 'numpy.float64'>"], "InitialAngularAcceleration must be either a float or an int."

	InitialAngularJerk = kwargs.get(
		"InitialAngularJerk",
		d3r(0)
	) # 0 or d3r(0)
	assert str(type(InitialAngularJerk)) in ["<class 'float'>","<class 'int'>","<class 'numpy.float64'>"], "InitialAngularJerk must be either a float or an int."

	InitialAngularSnap = kwargs.get(
		"InitialAngularSnap",
		d4r(0)
	) # 0 or d4r(0)
	assert str(type(InitialAngularSnap)) in ["<class 'float'>","<class 'int'>","<class 'numpy.float64'>"], "InitialAngularSnap must be either a float or an int."

	InitialTensionVelocity = kwargs.get(
        "InitialTensionVelocity",
        return_initial_tension_velocity(
			InitialTension,
			X_o,
			InitialAngularAcceleration=InitialAngularAcceleration,
			InitialAngularJerk=InitialAngularJerk
		)
    )
	assert np.shape(InitialTensionVelocity)==(2,) \
			and str(type(InitialTensionVelocity))=="<class 'numpy.ndarray'>", \
		"InitialTensionVelocity must be a numpy array of shape (2,)"

	InitialTensionAcceleration = kwargs.get(
	    "InitialTensionAcceleration",
	    return_initial_tension_acceleration(
	        InitialTension,
			InitialTensionVelocity,
	        X_o,
	        InitialAngularAcceleration=InitialAngularAcceleration,
			InitialAngularJerk=InitialAngularJerk,
			InitialAngularSnap=InitialAngularSnap
        )
    )
	assert np.shape(InitialTensionAcceleration)==(2,) \
			and str(type(InitialTensionAcceleration))=="<class 'numpy.ndarray'>", \
		"InitialTensionAcceleration must be a numpy array of shape (2,)"

	X_o = [X_o[0],X_o[1],InitialTension[0],InitialTension[1]]
	vm1 = (
		(
			BIC.v_MTU(X_o) * (
				1
				- np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT))
			)
			- (
				BIC.lTo*InitialTensionVelocity[0]
				/ (BIC.F_MAX*BIC.cT*np.cos(BIC.pa))
			)
		)
		/ (
			1
			- np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT))
		)
	)

	vm2 = (
		(
			TRI.v_MTU(X_o) * (
				1
				- np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT))
			)
			- (
				TRI.lTo*InitialTensionVelocity[1]
				/ (TRI.F_MAX*TRI.cT*np.cos(TRI.pa))
			)
		)
		/ (
			1
			- np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT))
		)
	)

	am1 = (
		(
			InitialTensionVelocity[0] * (
				np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT))
				* (
					BIC.v_MTU(X_o)
					- np.cos(BIC.pa)*vm1
				)
				/ (BIC.lTo*BIC.kT)
			)
			- InitialTensionAcceleration[0]
			+ BIC.F_MAX*BIC.cT*BIC.a_MTU(X_o) * (
				1
				- np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT))
			) / BIC.lTo
		)
		/ (
			BIC.F_MAX*BIC.cT * (
				1
				- np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT))
			) / (BIC.lTo*np.cos(BIC.pa))
		)
	)

	am2 = (
		(
			InitialTensionVelocity[1] * (
				np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT))
				* (
					TRI.v_MTU(X_o)
					- np.cos(TRI.pa)*vm2
				)
				/ (TRI.lTo*TRI.kT)
			)
			- InitialTensionAcceleration[1]
			+ TRI.F_MAX*TRI.cT*TRI.a_MTU(X_o) * (
				1
				- np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT))
			) / TRI.lTo
		)
		/ (
			TRI.F_MAX*TRI.cT * (
				1
				- np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT))
			) / (TRI.lTo*np.cos(TRI.pa))
		)
	)
	u1_func = lambda l: (
		(
			InitialTension[0]*np.cos(BIC.pa)
			+ BIC.m*vm1**2*np.tan(BIC.pa)**2/l
			- BIC.m*am1
			+ BIC.bm*BIC.F_MAX*np.cos(BIC.pa)**2*vm1
			- BIC.F_PE1(l,vm1)*BIC.F_MAX*np.cos(BIC.pa)**2
		)
		/ (BIC.FLV(l,vm1)*BIC.F_MAX*np.cos(BIC.pa)**2)
	)
	u2_func = lambda l: (
		(
			InitialTension[1]*np.cos(TRI.pa)
			+ TRI.m*vm2**2*np.tan(TRI.pa)**2/l
			- TRI.m*am2
			+ TRI.bm*TRI.F_MAX*np.cos(TRI.pa)**2*vm2
			- TRI.F_PE1(l,vm2)*TRI.F_MAX*np.cos(TRI.pa)**2
		)
		/ (TRI.FLV(l,vm2)*TRI.F_MAX*np.cos(TRI.pa)**2)
	)

	L1_UB = fsolve(u1_func,1.5*BIC.lo)
	L2_UB = fsolve(u2_func,1.5*TRI.lo)

	L1_LB = 0.75*BIC.lo
	if L1_UB > 1.25*BIC.lo:
		L1_UB = 1.25*BIC.lo
	L1 = np.linspace(L1_LB, L1_UB, 1001)
	U1 = np.array(list(map(u1_func,L1)))

	L2_LB = 0.75*TRI.lo
	if L2_UB > 1.25*TRI.lo:
		L2_UB = 1.25*TRI.lo
	L2 = np.linspace(L2_LB, L2_UB, 1001)
	U2 = np.array(list(map(u2_func,L2)))

	if PlotBool == True:
		plt.figure(figsize=(10,8))
		plt.title(r"Viable Initial $l_{m,1}$ and $u_{1}$ Values")
		plt.xlabel(r"$l_{m,1}$ (m)",fontsize=14)
		plt.ylabel(r"$u_{1}$",fontsize=14)
		plt.scatter(L1,U1)
		plt.plot([BIC.lo,BIC.lo],[0,1],'0.70',linestyle='--')
		plt.gca().set_ylim((0,1))
		plt.gca().set_xticks(
			[0.25*BIC.lo,
			0.5*BIC.lo,
			0.75*BIC.lo,
			BIC.lo,
			1.25*BIC.lo,
			1.5*BIC.lo,
			1.75*BIC.lo]
			)
		plt.gca().set_xticklabels(
			["",
			r"$\frac{1}{2}$ $l_{o,2}$",
			"",
			r"$l_{o,2}$",
			"",
			r"$\frac{3}{2}$ $l_{o,2}$",
			""],
			fontsize=12)

		plt.figure(figsize=(10,8))
		plt.title(r"Viable Initial $l_{m,2}$ and $u_{2}$ Values")
		plt.xlabel(r"$l_{m,2}$ (m)",fontsize=14)
		plt.ylabel(r"$u_{2}$",fontsize=14)
		plt.scatter(L2,U2)
		plt.plot([TRI.lo,TRI.lo],[0,1],'0.70',linestyle='--')
		plt.gca().set_ylim((0,1))
		plt.gca().set_xticks(
			[0.25*TRI.lo,
			0.5*TRI.lo,
			0.75*TRI.lo,
			TRI.lo,
			1.25*TRI.lo,
			1.5*TRI.lo,
			1.75*TRI.lo]
			)
		plt.gca().set_xticklabels(
			["",
			r"$\frac{1}{2}$ $l_{o,2}$",
			"",
			r"$l_{o,2}$",
			"",
			r"$\frac{3}{2}$ $l_{o,2}$",
			""],
			fontsize=12)

		plt.show()
	return(L1,U1,L2,U2)
#### NEED TO RETURN INITIAL TENSIONS BEFORE CALLING THIS FUNCTION
def return_initial_activations_given_muscle_lengths_and_tendon_tensions(InitialTension,L_m,X_o,**kwargs):
	"""
	This function returns initial muscle activations for a fixed muscle length and a given pretensioning level, as derived from (***insert file_name here for scratchwork***) for the system that starts from rest. (Ex. pendulum_eqns.reference_trajectories._01).

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**kwargs

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Seed - Can see the random tension generated. When FixedInitialTension is provided, this seed will apply only to the initial conditions for activation and muscle length.

	2) InitialTensionAcceleration - must be a numpy array of shape (2,). Default is set to the value generated from joint angle IC's for pendulum_eqns.reference_trajectories._01.py (B+A,0,-Aw²,0,Aw⁴,etc.). If using a different reference trajectory, it would be best to either set all joint angle IC's to zero (like for pendulum_eqns.reference_trajectories._02.py) or to derive the value and pass it through the kwargs here (NOTE: must also consider how changing angular IC's effect other state derivative IC's).

	3) InitialAngularAcceleration - must be either a numpy.float64, float, or int. Default is 0. Choice of reference trajectory *should* not matter as it is either 0 or d2r(0) (either by convention or by choice).

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""
	PlotBool = kwargs.get("PlotBool",False)
	assert type(PlotBool)==bool,"PlotBool must be a boolean. Default is False."

	InitialAngularAcceleration = kwargs.get(
	            "InitialAngularAcceleration",
	            0
	            ) # 0 or d2r(0)
	assert str(type(InitialAngularAcceleration)) in ["<class 'float'>","<class 'int'>","<class 'numpy.float64'>"], "InitialAngularAcceleration must be either a float or an int."

	InitialTensionAcceleration = kwargs.get(
	            "InitialTensionAcceleration",
	            return_initial_tension_acceleration(
	                InitialTension,
	                X_o,
	                InitialAngularAcceleration=InitialAngularAcceleration
	                )
	            )
	assert np.shape(InitialTensionAcceleration)==(2,) \
			and str(type(InitialTensionAcceleration))=="<class 'numpy.ndarray'>", \
		"InitialTensionAcceleration must be a numpy array of shape (2,)"

	assert len(L_m)==2 and type(L_m)==list, \
		"L_m must be a list of length 2."

	L1_UB = BIC.lo*BIC.L_CE_max*(
		BIC.k_1*np.log(
			np.exp(
				(
					BIC.m*InitialTensionAcceleration[0]
					+ (BIC.F_MAX*BIC.cT/BIC.lTo)
					* (
						1
						- np.exp(
							-InitialTension[0]
							/ (BIC.F_MAX*BIC.cT*BIC.kT)
						)
					)
					* (BIC.C1*InitialTension[0]- BIC.m*BIC.a_MTU(X_o))
				)
				/ (
					BIC.F_MAX*BIC.C1**2
					* BIC.c_1*BIC.k_1
					* (BIC.F_MAX*BIC.cT/BIC.lTo)
					* (
						1
						- np.exp(
							-InitialTension[0]
							/ (BIC.F_MAX*BIC.cT*BIC.kT)
						)
					)
				)
			)
			- 1
		)
		+ BIC.Lr1
	)
	L2_UB = TRI.lo*TRI.L_CE_max*(
		TRI.k_1*np.log(
			np.exp(
				(
					TRI.m*InitialTensionAcceleration[1]
					+ (TRI.F_MAX*TRI.cT/TRI.lTo)
					* (
						1
						- np.exp(
							-InitialTension[1]
							/ (TRI.F_MAX*TRI.cT*TRI.kT)
						)
					)
					* (TRI.C1*InitialTension[1] - TRI.m*TRI.a_MTU(X_o))
				)
				/ (
					TRI.F_MAX*TRI.C1**2
					* TRI.c_1*TRI.k_1
					* (TRI.F_MAX*TRI.cT/TRI.lTo)
					* (
						1
						- np.exp(
							-InitialTension[1]
							/ (TRI.F_MAX*TRI.cT*TRI.kT)
						)
					)
				)
			)
			- 1
		)
		+ TRI.Lr1
	)

	L1_LB = 0.5*BIC.lo
	if L1_UB > 1.5*BIC.lo:
		L1_UB = 1.5*BIC.lo

	assert L1_LB<=L_m[0]<=L1_UB, "For the chosen tension level, fixed muscle length produces infeasible activations."

	U1 = (
		(
			BIC.m*InitialTensionAcceleration[0]
			+ (BIC.F_MAX*BIC.cT/BIC.lTo)
			* (1-np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT)))
			* (
				BIC.C1*InitialTension[0]
				- BIC.m*BIC.a_MTU(X_o)
				- BIC.F_MAX*BIC.C1**3
				* BIC.c_1*BIC.k_1
				* np.log(
					np.exp(
						(L_m[0]/(BIC.lo*BIC.L_CE_max) - BIC.Lr1)
						/ BIC.k_1
					)
					+ 1
				)
			)
		)
		/ (
			BIC.F_MAX*BIC.C1**2
			* (BIC.F_MAX*BIC.cT/BIC.lTo)
			* (1-np.exp(-InitialTension[0]/(BIC.F_MAX*BIC.cT*BIC.kT)))
			* np.exp(-(abs((L_m[0]-BIC.lo)/(BIC.lo*BIC.omega))**BIC.rho))
		)
	)
	assert 0<=U1<=1, "U1 should be in [0,1)"

	L2_LB = 0.5*TRI.lo
	if L2_UB > 1.5*TRI.lo:
		L2_UB = 1.5*TRI.lo

	assert L2_LB<=L_m[1]<=L2_UB, "For the chosen tension level, fixed muscle length produces infeasible activations."

	U2 = (
		(
			TRI.m*InitialTensionAcceleration[1]
			+ (TRI.F_MAX*TRI.cT/TRI.lTo)
			* (1-np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT)))
			* (
				TRI.C1*InitialTension[1]
				- TRI.m*TRI.a_MTU(X_o)
				- TRI.F_MAX*TRI.C1**3
				* TRI.c_1*TRI.k_1
				* np.log(
					np.exp(
						(L_m[1]/(TRI.lo*TRI.L_CE_max) - TRI.Lr1)
						/ TRI.k_1
					)
					+ 1
				)
			)
		)
		/ (
			TRI.F_MAX*TRI.C1**2
			* (TRI.F_MAX*TRI.cT/TRI.lTo)
			* (1-np.exp(-InitialTension[1]/(TRI.F_MAX*TRI.cT*TRI.kT)))
			* np.exp(-(abs((L_m[1]-TRI.lo)/(TRI.lo*TRI.omega))**TRI.rho))
		)
	)
	assert 0<=U2<=1, "U2 should be in [0,1)"
	return(L_m[0],U1,L_m[1],U2)

def find_viable_initial_values(**kwargs):
	"""
	This function returns initial conditions for the system that starts from rest. (Ex. pendulum_eqns.reference_trajectories._01)

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**kwargs

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) FixedInitialTension - Must be a (2,1) numpy.ndarray. Run find_initial_tension outside of the loop for a given seed and then feed it through the pipeline.

	2) ReturnAll - Can return all initial values for a given tension level. Will be fed through to return_random_initial_muscle_lengths_and_activations.

	3) Seed - Can see the random tension generated. When FixedInitialTension is provided, this seed will apply only to the initial conditions for activation and muscle length.

	4) InitialAngularAcceleration - Will be passed to return_random_initial_muscle_lengths_and_activations(), must be a scalar, int or numpy.float64.

	5) InitialTensionAcceleration - Will be passed to return_random_initial_muscle_lengths_and_activations(), must be a numpy array of shape (2,).

	6) FixedInitialMuscleLengths - must be a list of length 2 or None (Default). If is None, then program will assign this value randomly. Used for trials where we wish to hold muscle length constant for different tension levels.
	"""
	FixedInitialTension = kwargs.get("FixedInitialTension",None)
	assert (FixedInitialTension is None) or \
			(str(type(FixedInitialTension)) == "<class 'numpy.ndarray'>"
			and np.shape(FixedInitialTension) == (2,1)),\
		(
		"FixedInitialTension must either be None (Default) or a (2,1) numpy.ndarray."
		+ "\nCurrent type: "
		+ str(type(FixedInitialTension))
		+ "\nCurrent shape: "
		+ str(np.shape(FixedInitialTension))
		)

	FixedInitialMuscleLengths = kwargs.get("FixedInitialMuscleLengths",None)
	if FixedInitialMuscleLengths is not None:
	    assert type(FixedInitialMuscleLengths)==list and \
	            len(FixedInitialMuscleLengths)==2, \
	        "FixedInitialMuscleLengths should be either None (Default) or a list of length 2."

	ReturnAll = kwargs.get("ReturnAll",False)
	assert type(ReturnAll)==bool, "ReturnAll must be a bool."

	Seed = kwargs.get("Seed",None)
	assert type(Seed) in [float,int] or Seed is None, "Seed must be a float or an int or None."
	np.random.seed(Seed)

	X_o = np.array([r(0),dr(0)])
	if FixedInitialTension is None:
		T = return_initial_tension(X_o,**kwargs)
	else:
		T = FixedInitialTension

	if FixedInitialMuscleLengths is None:
		L1,U1,L2,U2 = return_random_initial_muscle_lengths_and_activations(T,X_o,**kwargs)
		PositiveActivations = False
		while PositiveActivations == False:
			rand_index = np.random.choice(len(L1),2)
			u1,u2 = U1[rand_index[0]],U2[rand_index[1]]
			if u1>0 and u2>0:
				PositiveActivations = True
		# import ipdb; ipdb.set_trace()
		if ReturnAll==False:
			# return(
			# 	T,
			# 	np.array([L1[rand_index[0]],L2[rand_index[1]]]),
			# 	np.array([U1[rand_index[0]],U2[rand_index[1]]])
			# 	)
			l1_index = np.where((L1-0.8*BIC.lo)**2==min((L1-0.8*BIC.lo)**2))[0][0]
			l2_index = np.where((L2-1.2*TRI.lo)**2==min((L2-1.2*TRI.lo)**2))[0][0]
			# import ipdb; ipdb.set_trace()
			return(
				T,
				np.array([L1[l1_index],L2[l2_index]]),
				np.array([U1[l1_index],U2[l2_index]])
				)
		else:
			return(T,L1,L2,U1,U2)

	else:
		L1,U1,L2,U2 = return_initial_activations_given_muscle_lengths_and_tendon_tensions(T,FixedInitialMuscleLengths,X_o,**kwargs)

		assert ReturnAll != True, "For fixed initial muscle lengths, ReturnAll must be False as the function return_initial_activations_given_muscle_lengths_and_tendon_tensions only returns single values."

		return(
			T,
			np.array([L1,L2]),
			np.array([U1,U2])
			)
