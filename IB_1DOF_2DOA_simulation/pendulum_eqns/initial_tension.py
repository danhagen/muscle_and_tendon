from pendulum_eqns.state_equations import *
from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.reference_trajectories._01 import *

if params["g"] == 0:
	MaxStep_Tension = 0.20*min(BIC.F_MAX,TRI.F_MAX) # percentage of positive maximum.
	Tension_Bounds = [[0,BIC.F_MAX],[0,TRI.F_MAX]]
else:
	MaxStep_Tension = 0.01**min(BIC.F_MAX,TRI.F_MAX) # percentage of positive maximum.
	Tension_Bounds = [[0,BIC.F_MAX],[0,0.30*TRI.F_MAX]]

def return_initial_tension(X_o,**kwargs):
	"""
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Takes in initial state numpy.ndarray (X_o) of shape (2,) and returns an initial tension (2,) numpy.ndarray. InitialAngularAcceleration should be chosen or left to default to ensure proper IC's.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**kwargs

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Tension_Bounds.

	2) InitialAngularAcceleration - must be a float or an int. Default is 0 (starting from rest).

	3) ReturnMultipleInitialTensions - must be either True or False. Default is False.

	4) Seed - must be a float or an int. Default is None (seeded by current time).

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	"""
	ReturnMultipleInitialTensions = kwargs.get("ReturnMultipleInitialTensions",False)
	assert type(ReturnMultipleInitialTensions)==bool,\
	 		"ReturnMultipleInitialTensions must be either True or False. Default is False."

	Seed = kwargs.get("Seed",None)
	assert type(Seed) in [float,int] or Seed is None, "Seed must be a float or an int or None."
	np.random.seed(Seed)

	assert (np.shape(X_o) in [(2,),(4,),(8,)]) \
			and (str(type(X_o)) == "<class 'numpy.ndarray'>"), \
		"X_o must be a (2,), (4,), or (8,) numpy.ndarray."

	InitialAngularAcceleration = kwargs.get("InitialAngularAcceleration",0) # or d2r(0)
	assert str(type(InitialAngularAcceleration)) in ["<class 'float'>","<class 'int'>","<class 'numpy.float64'>"], "InitialAngularAcceleration must be either a float or an int."

	Bounds = kwargs.get("Bounds",Tension_Bounds)
	assert type(Bounds) == list and np.shape(Bounds) == (2,2), "Bounds for Tension Control must be a (2,2) list."
	assert Bounds[0][0]<Bounds[0][1],"Each set of bounds must be in ascending order."
	assert Bounds[1][0]<Bounds[1][1],"Each set of bounds must be in ascending order."

	Return_k = kwargs.get("Return_k",False)
	assert type(Return_k)==bool, "Return_k must be either true or false (default)."

	Constraint = lambda T1: (
		(
			(params["M"]*params["L"]**2/3)*InitialAngularAcceleration
			+ params["M"]*params["g"]*params["L"]*np.sin(X_o[0])/2
			- BIC.R(X_o[0])*T1
		)
		/ TRI.R(X_o[0])
	) # Returns T2 given T1
	InverseConstraint = lambda T2: (
		(
			(params["M"]*params["L"]**2/3)*InitialAngularAcceleration
			+ params["M"]*params["g"]*params["L"]*np.sin(X_o[0])/2
			- TRI.R(X_o[0])*T2
		)
		/ BIC.R(X_o[0])
	) # Returns T1 given T2
	LowerBound_x = max(Bounds[0][0],InverseConstraint(Bounds[1][0]))
	LowerBound_y = Constraint(LowerBound_x)
	UpperBound_x = min(Bounds[0][1],InverseConstraint(Bounds[1][1]))
	UpperBound_y = Constraint(UpperBound_x)

	LowerBoundVector = np.array([[LowerBound_x],[LowerBound_y]])
	UpperBoundVector = np.array([[UpperBound_x],[UpperBound_y]])

	if ReturnMultipleInitialTensions == True:
		k_array = np.linspace(0.05,1,8)
		InitialTension = [
				(UpperBoundVector-LowerBoundVector)*k + LowerBoundVector
				for k in k_array
		]
	else:
		k = np.random.rand()
		InitialTension = (UpperBoundVector-LowerBoundVector)*k + LowerBoundVector
	if Return_k==False:
		return(InitialTension)
	else:
		if ReturnMultipleInitialTensions==True:
			return(InitialTension,k_array)
		else:
			return(InitialTension,k)

def return_initial_tension_velocity(T_o,X_o,**kwargs):
	"""
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Takes in initial tension (T_o) of shape (2,), initial state numpy.ndarray (X_o) of shape (2,),  and returns an initial tension acceleration of shape (2,). InitialAngularAcceleration and InitialAngularSnap should be chosen or left to default to ensure proper IC's. Default conditions will return zero tension acceleration (i.e., starting from rest), unless otherwise dictated by the particular solution.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**kwargs

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Tension_Bounds.

	2) InitialAngularAcceleration - must be a numpy.float64, float, or an int. Default is 0 (starting from rest).

	3) InitialAngularJerk - must be a numpy.float64, float, or an int. Default is 0 (starting from rest).

	4) ReturnMultipleInitialTensions - must be either True or False. Default is False.

	5) Seed - must be a float or an int. Default is None (seeded by current time).

	6) ParticularSolution - Must be a numpy array of shape (2,). Default is numpy.array([0,0]). Must be in the nullspace of [BIC.R(X_o[0]),TRI.R(X_o[0])].

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	"""
	InitialAngularJerk = kwargs.get("InitialAngularJerk",0)
	assert str(type(InitialAngularJerk)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"], "InitialAngularJerk must be either a float, int or numpy.float64"

	InitialAngularAcceleration = kwargs.get("InitialAngularAcceleration",0)
	assert str(type(InitialAngularAcceleration)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"], "InitialAngularAcceleration must be either a float, int or numpy.float64"

	ParticularSolution = kwargs.get("ParticularSolution",np.array([0,0]))
	assert np.shape(ParticularSolution)==(2,) and str(type(ParticularSolution))=="<class 'numpy.ndarray'>", "ParticularSolution must be a (2,) numpy.ndarray"
	assert abs(BIC.R(X_o[0])*ParticularSolution[0]
				+ TRI.R(X_o[0])*ParticularSolution[1]) < 1e-6, \
		"ParticularSolution must be in the nullspace of [BIC.R,TRI.R]."

	HomogeneousSolution = (
		(
			(
				(params["M"]*params["L"]**2/3)*InitialAngularJerk
				+ params["M"]*params["g"]*params["L"]*X_o[1]*np.cos(X_o[0])/2
				- X_o[1] * (
					BIC.dR(X_o[0])*T_o[0]
					+ TRI.dR(X_o[0])*T_o[1]
				)
			)
			/ (BIC.R(X_o[0])**2 + TRI.R(X_o[0])**2)
		)
		* np.array([BIC.R(X_o[0]),TRI.R(X_o[0])])
	)

	return(ParticularSolution+HomogeneousSolution)

def return_initial_tension_acceleration(T_o,dT_o,X_o,**kwargs):
	"""
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Takes in initial tension (T_o) of shape (2,), initial tension velocity (dT_o) of shape (2,), initial state numpy.ndarray (X_o) of shape (2,),  and returns an initial tension acceleration of shape (2,). InitialAngularAcceleration and InitialAngularSnap should be chosen or left to default to ensure proper IC's. Default conditions will return zero tension acceleration (i.e., starting from rest), unless otherwise dictated by the particular solution.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**kwargs

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Tension_Bounds.

	2) InitialAngularAcceleration - must be a numpy.float64, float, or an int. Default is 0 (starting from rest).

	3) InitialAngularJerk - must be a numpy.float64, float, or an int. Default is 0 (starting from rest).

	4) InitialAngularSnap - must be a numpy.float64, float, or an int. Default is 0 (starting from rest).

	5) ReturnMultipleInitialTensions - must be either True or False. Default is False.

	6) Seed - must be a float or an int. Default is None (seeded by current time).

	7) ParticularSolution - Must be a numpy array of shape (2,). Default is numpy.array([0,0]). Must be in the nullspace of [BIC.R(X_o[0]),TRI.R(X_o[0])].

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	"""
	InitialAngularSnap = kwargs.get("InitialAngularSnap",0)
	assert str(type(InitialAngularSnap)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"], "InitialAngularSnap must be either a float, int or numpy.float64"

	InitialAngularJerk = kwargs.get("InitialAngularJerk",0)
	assert str(type(InitialAngularJerk)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"], "InitialAngularJerk must be either a float, int or numpy.float64"

	InitialAngularAcceleration = kwargs.get("InitialAngularAcceleration",0)
	assert str(type(InitialAngularAcceleration)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"], "InitialAngularAcceleration must be either a float, int or numpy.float64"

	ParticularSolution = kwargs.get("ParticularSolution",np.array([0,0]))
	assert np.shape(ParticularSolution)==(2,) and str(type(ParticularSolution))=="<class 'numpy.ndarray'>", "ParticularSolution must be a (2,) numpy.ndarray"
	assert abs(BIC.R(X_o[0])*ParticularSolution[0]
				+ TRI.R(X_o[0])*ParticularSolution[1]) < 1e-6, \
		"ParticularSolution must be in the nullspace of [BIC.R,TRI.R]."

	HomogeneousSolution = (
		(
			(
				(params["M"]*params["L"]**2/3)*InitialAngularSnap
				+ params["M"]*params["g"]*params["L"]*(
					InitialAngularAcceleration*np.cos(X_o[0])
					- X_o[1]**2*np.sin(X_o[0])
				)/2
				- X_o[1]**2 * (
					BIC.d2R(X_o[0])*T_o[0]
					+ TRI.d2R(X_o[0])*T_o[1]
				)
				- InitialAngularAcceleration * (
					BIC.dR(X_o[0])*T_o[0]
					+ TRI.dR(X_o[0])*T_o[1]
				)
				- 2*X_o[1] * (
					BIC.dR(X_o[0])*dT_o[0]
					+ TRI.dR(X_o[0])*dT_o[1]
				)
			)
			/ (BIC.R(X_o[0])**2 + TRI.R(X_o[0])**2)
		)
		* np.array([BIC.R(X_o[0]),TRI.R(X_o[0])])
	)
	return(ParticularSolution+HomogeneousSolution)

def plot_initial_tension_values(X_o,**kwargs):
	"""
	Takes in initial state numpy.ndarray (X_o) of shape (2,) and plots 1000 initial tension values (2,).

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	**kwargs
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Seed - must be a scalar value. Default is None.

	2) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Tension_Bounds.

	3) InitialAngularAcceleration - must be a float or an int. Default is 0 (starting from rest).

	4) NumberOfPoints - must be an int. Default is 1000.

	"""

	assert (np.shape(X_o) in [(2,),(4,),(8,)]) \
			and (str(type(X_o)) == "<class 'numpy.ndarray'>"), \
		"X_o must be a (2,), (4,), or (8,) numpy.ndarray."

	InitialAngularAcceleration = kwargs.get("InitialAngularAcceleration",0) # or d2r(0)
	assert type(InitialAngularAcceleration) in [float,int], "InitialAngularAcceleration must be either a float or an int."

	Seed = kwargs.get("Seed",None)
	assert type(Seed) in [float,int] or Seed is None, "Seed must be a float or an int or None."
	np.random.seed(Seed)

	Bounds = kwargs.get("Bounds",Tension_Bounds)
	assert type(Bounds) == list and np.shape(Bounds) == (2,2), "Bounds for Tension Control must be a (2,2) list."
	assert Bounds[0][0]<Bounds[0][1],"Each set of bounds must be in ascending order."
	assert Bounds[1][0]<Bounds[1][1],"Each set of bounds must be in ascending order."

	NumberOfPoints = kwargs.get("NumberOfPoints",1000)
	assert type(NumberOfPoints) == int, "NumberOfPoints must be an int."

	fig = plt.figure(figsize=(10,8))
	ax1 = plt.gca()

	DescriptiveTitle = "Plotting Randomly Generated\nInitial Tendon Tensions"

	ax1.set_title(DescriptiveTitle,Fontsize=20,y=1)

	#Bound Constraints
	ax1.plot([Bounds[0][0],Bounds[0][1]],[Bounds[1][0],Bounds[1][0]],'k--')
	ax1.plot([Bounds[0][0],Bounds[0][1]],[Bounds[1][1],Bounds[1][1]],'k--')
	ax1.plot([Bounds[0][0],Bounds[0][0]],[Bounds[1][0],Bounds[1][1]],'k--')
	ax1.plot([Bounds[0][1],Bounds[0][1]],[Bounds[1][0],Bounds[1][1]],'k--')

	Constraint = lambda T1: -(BIC.R(X_o[0])*T1 + (params["M"]*params["L"]**2/3)*InitialAngularAcceleration)/TRI.R(X_o[0])
	Input1 = np.linspace(
		Bounds[0][0]-0.2*np.diff(Bounds[0]),
		Bounds[0][1]+0.2*np.diff(Bounds[0]),
		1001)
	Input2 = Constraint(Input1)
	ax1.plot(Input1,Input2,'r',lw=2)
	statusbar = dsb(0,NumberOfPoints,title=plot_initial_tension_values.__name__)
	for i in range(NumberOfPoints):
		T = return_initial_tension(X_o,**kwargs)
		ax1.plot(T[0],T[1],'go')
		statusbar.update(i)
	ax1.set_xlabel(r'$T_{1}$',fontsize=14)
	ax1.set_ylabel(r'$T_{2}$',fontsize=14)
	ax1.set_xlim([Input1[0],Input1[-1]])
	ax1.set_ylim([Input2[0],Input2[-1]])
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.set_aspect('equal')
	plt.show()
