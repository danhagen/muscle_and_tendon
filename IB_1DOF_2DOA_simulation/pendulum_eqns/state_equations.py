from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.reference_trajectories._01 import *
from danpy.useful_functions import *
from danpy.sb import *
import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

"""
################################
########## Parameters ##########
################################

c_{1} &= -\frac{3g}{2L} \\
c_{2} &= \frac{3}{ML^2} \\
c_{3} &= \cos(\rho_1) \\
c_{4} &= \cos(\rho_2)
c_{5} &= \frac{\cos(\alpha_{1})}{m_1} \\
c_{6} &= \frac{\cos^2(\alpha_{1})}{m_1} \\
c_{7} &= \frac{b_{m,1}\cos^2(\alpha_{1})}{m_1} \\
c_{8} &= \tan^2(\alpha_{1}) \\
c_{9} &= \frac{\cos(\alpha_{2})}{m_2} \\
c_{10} &= \frac{\cos^2(\alpha_{2})}{m_2} \\
c_{11} &= \frac{b_{m,2}\cos^2(\alpha_{2})}{m_2} \\
c_{12} &= \tan^2(\alpha_{2}) \\

################################
######## Tension Driven ########
################################

\dot{x}_1 &= x_{2} \\
\dot{x}_2 &= c_{1}\sin(x_{1}) + c_{2}R_{1}u_{1} - c_{2}R_{2}u_{2} \\
u_1 &= T_{1} \\
u_2 &= T_{2} \\

################################
#### Muscle Velocity Driven ####
################################

\dot{x}_1 &= x_{2} \\
\dot{x}_2 &= c_{1}\sin(x_{1}) + c_{2}R_{1}x_{3} - c_{2}R_{2}x_{4} \\
\dot{x}_3 &= K_{T,1}(v_{MTU,1} - c_{3}u_1) \\
\dot{x}_4 &= K_{T,2}(v_{MTU,2} - c_{4}u_2) \\
u_1 &= \dot{l}_{m,1} \\
u_2 &= \dot{l}_{m,2} \\

################################
### Muscle Activation Driven ###
################################

\dot{x}_1 &= x_{2} \\
\dot{x}_2 &= c_{1}\sin(x_{1}) + c_{2}R_{1}x_{3} - c_{2}R_{2}x_{4} \\
\dot{x}_3 &= K_{T,1}(v_{MTU,1} - c_{3}x_7) \\
\dot{x}_4 &= K_{T,2}(v_{MTU,2} - c_{4}x_8) \\
\dot{x}_5 &= x_7 \\
\dot{x}_6 &= x_8 \\
\dot{x}_7 &= c_5x_3 - c_6F_{PE,1}(x_5,x_7) - c_7x_7 + \frac{c_{8}x_7^2}{x_5} - c_6F_{LV,1}(x_5,x_7)u_1 \\
\dot{x}_8 &= c_9x_4 - c_{10}F_{PE,2}(x_6,x_8) - c_{11}x_8 + \frac{c_{12}x_8^2}{x_6} - c_{10}F_{LV,2}(x_6,x_8)u_2 \\
u_1 &= \alpha_{1} \\
u_2 &= \alpha_{2} \\

"""

params["g"] = 9.80 # m/s²
params["L"] = 0.45 # m
params["M"] = 1.6 # kg

params["Amp"] = Amp
params["Base"] = Base
params["Freq"] = Freq

params["InitialAngle"] = r(0)
params["InitialAngularVelocity"] = dr(0) # or 0 (from rest)
params["InitialAngularAcceleration"] = d2r(0) # or 0 (from rest)
params["InitialAngularJerk"] = d3r(0) # or 0 (from rest)
params["InitialAngularSnap"] = d4r(0) # or 0 (from rest)

if params["g"] == 0: # Horizontal plane
	params["Tension Bounds"] = [[0,BIC.F_MAX],[0,TRI.F_MAX]]
else: # Vertical plane
	params["Tension Bounds"] = [[0,BIC.F_MAX],[0,0.30*TRI.F_MAX]]

if params["g"] == 0:
	MaxStep_Tension = 0.20*min(BIC.F_MAX,TRI.F_MAX) # percentage of positive maximum.
	Tension_Bounds = [[0,BIC.F_MAX],[0,TRI.F_MAX]]
else:
	MaxStep_Tension = 0.01**min(BIC.F_MAX,TRI.F_MAX) # percentage of positive maximum.
	Tension_Bounds = [[0,BIC.F_MAX],[0,0.30*TRI.F_MAX]]

BIC.C1 = np.cos(BIC.pa)
BIC.C2 = np.cos(BIC.pa)/BIC.m
BIC.C3 = BIC.F_MAX*np.cos(BIC.pa)**2/BIC.m
BIC.C4 = BIC.F_MAX*BIC.bm*np.cos(BIC.pa)**2/(BIC.m*BIC.lo)
BIC.C5 = np.tan(BIC.pa)**2

TRI.C1 = np.cos(TRI.pa)
TRI.C2 = np.cos(TRI.pa)/TRI.m
TRI.C3 = TRI.F_MAX*np.cos(TRI.pa)**2/TRI.m
TRI.C4 = TRI.F_MAX*TRI.bm*np.cos(TRI.pa)**2/(TRI.m*TRI.lo)
TRI.C5 = np.tan(TRI.pa)**2

##########################
### Pendulum Equations ###
##########################

def dX1_dt(X):
	return(X[1])
def dX2_dt(X):
	return(
		(
			-(params["M"]*params["g"]*params["L"]/2)*np.sin(X[0])
			+ BIC.R(X[0])*X[2]
			+ TRI.R(X[0])*X[3]
		) / (params["M"]*params["L"]**2/3)
	)

##########################
##### Update Classes #####
#####  with v/a_MTU  #####
##########################

BIC.add_v_MTU([dX1_dt,dX2_dt])
BIC.add_a_MTU([dX1_dt,dX2_dt])

TRI.add_v_MTU([dX1_dt,dX2_dt])
TRI.add_a_MTU([dX1_dt,dX2_dt])

##########################
##########################

k1,k2,k3,k4 = 100,100,100,10
class Pendulum_1DOF_2DOA:
	def __init__(self,m1,m2,Time,**params):
		self.X = np.zeros((8,len(Time)))
		# self.X[:,0] = X_o

		self.U = np.zeros((2,len(Time)))
		# self.U[:,0] = U_o

		self.Time = Time
		self.dt = Time[1]-Time[0]

		self.M = params.get("M",1.6) # in kg
		is_number(self.M,type(self).__name__+"M",default=1.6,notes="Given in kg.")
		self.g = params.get("g",9.8) # m/s²
		is_number(self.g,type(self).__name__+"g",default=1.6,notes="Given in m/s².")
		self.L = params.get("L",0.45) # in m
		is_number(self.L,type(self).__name__+"L",default=1.6,notes="Given in m.")

		self.MgL_2 = self.M*self.g*self.L/2
		self.Inertia = self.M*self.L**2/3

		self.m1 = m1
		assert str(type(self.m1))=="<class 'pendulum_eqns.physiology.muscle_params_BIC_TRI.Musculotendon'>", \
				"m1 must be of type " + "<class 'pendulum_eqns.physiology.muscle_params_BIC_TRI.Musculotendon'>"
		self.m2 = m2
		assert str(type(self.m2))=="<class 'pendulum_eqns.physiology.muscle_params_BIC_TRI.Musculotendon'>", \
				"m2 must be of type " + "<class 'pendulum_eqns.physiology.muscle_params_BIC_TRI.Musculotendon'>"
		self.TensionBounds = params.get(
			"Tension Bounds",
			[[0,self.m1.F_MAX],[0,self.m2.F_MAX]]
		)
		assert np.shape(self.TensionBounds)==(2,2), "TensionBounds must be a (2,2) list"

		self.InitialAngle = params.get(
			"InitialAngle",
			r(0)
		) # in rad
		assert self.InitialAngle!=0,"Error here."
		is_number(
			self.InitialAngle,
			type(self).__name__+"InitialAngle",
			default=r(0),
			notes="Given in rad."
		)

		self.InitialAngularVelocity = params.get(
			"InitialAngularVelocity",
			dr(0)
		) # in rad⋅s⁻¹
		is_number(
			self.InitialAngularVelocity,
			type(self).__name__+"InitialAngularVelocity",
			default=dr(0),
			notes="Given in rad⋅s⁻¹."
		)

		self.InitialAngularAcceleration = params.get(
			"InitialAngularAcceleration",
			d2r(0)
		) # in rad⋅s⁻²
		assert self.InitialAngularAcceleration!=0,"Error here."
		is_number(
			self.InitialAngularAcceleration,
			type(self).__name__+"InitialAngularAcceleration",
			default=d2r(0),
			notes="Given in rad⋅s⁻²."
		)
		self.InitialAngularJerk = params.get(
			"InitialAngularJerk",
			d3r(0)
		) # in rad⋅s⁻³
		is_number(
			self.InitialAngularJerk,
			type(self).__name__+"InitialAngularJerk",
			default=d3r(0),
			notes="Given in rad⋅s⁻³."
		)
		self.InitialAngularSnap = params.get(
			"InitialAngularSnap",
			d4r(0)
		) # in rad⋅s⁻⁴
		is_number(
			self.InitialAngularSnap,
			type(self).__name__+"InitialAngularSnap",
			default=d4r(0),
			notes="Given in rad⋅s⁻⁴."
		)

	def find_initial_tension(self,**kwargs):
		"""
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Returns an initial tension (2,) numpy.ndarray.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) ReturnMultipleInitialTensions - must be either True or False. Default is False.

		2) Seed - must be a float or an int. Default is None (seeded by current time).

		3) Return - must be either true or false (default). When false, function will assign the initial tension value to self.

		4) Bounds - must be a list of size (2,2) or None (in which case the self.TensionBounds will be used)

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		"""
		ReturnMultipleInitialTensions = kwargs.get("ReturnMultipleInitialTensions",False)
		assert type(ReturnMultipleInitialTensions)==bool,\
		 		"ReturnMultipleInitialTensions must be either True or False. Default is False."

		Seed = kwargs.get("Seed",None)
		assert type(Seed) in [float,int] or Seed is None, "Seed must be a float or an int or None."
		np.random.seed(Seed)

		k = kwargs.get("k",None)
		if k is not None:
			is_number(k,"k")

		Return = kwargs.get("Return",False)
		assert type(Return)==bool, "Return must be either true or false (default). When false, initial tensions will be assigned to self."

		Return_k = kwargs.get("Return_k",False)
		assert type(Return_k)==bool, "Return_k must be either true or false (default)."

		Bounds = kwargs.get("Bounds",None)
		if Bounds is None:
			Bounds = self.TensionBounds
		else:
			assert np.shape(Bounds)==(2,2) and type(Bounds)==list, \
				"Bounds must be a list of shape (2,2)."

		Constraint = lambda T1: (
			(
				self.Inertia*self.InitialAngularAcceleration
				+ self.MgL_2*np.sin(self.InitialAngle)
				- self.m1.R(self.InitialAngle)*T1
			)
			/ self.m2.R(self.InitialAngle)
		) # Returns T2 given T1
		InverseConstraint = lambda T2: (
			(
				self.Inertia*self.InitialAngularAcceleration
				+ self.MgL_2*np.sin(self.InitialAngle)
				- self.m2.R(self.InitialAngle)*T2
			)
			/ self.m1.R(self.InitialAngle)
		) # Returns T1 given T2
		LowerBound_x = max(
			Bounds[0][0],
			InverseConstraint(Bounds[1][0])
		)
		LowerBound_y = Constraint(LowerBound_x)
		UpperBound_x = min(
			Bounds[0][1],
			InverseConstraint(Bounds[1][1])
		)
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
			if k is None:
				k = np.random.rand()
			InitialTension = (
				(UpperBoundVector-LowerBoundVector)*k
				+ LowerBoundVector
			)

		if Return==True:
			if Return_k==False:
				return(InitialTension)
			else:
				if ReturnMultipleInitialTensions==True:
					return(InitialTension,k_array)
				else:
					return(InitialTension,k)
		else:
			if ReturnMultipleInitialTensions==True:
				self.k = k_array
				self.InitialTensionsArray = InitialTension # in this case we need to run self.choose_initial_tension(self,i) where i is the index of the InitialTension in self.InitialTensions.
			else:
				self.k = k
				self.T_o = InitialTension
	def choose_initial_tension(self,i):
		assert hasattr(self,'InitialTensionsArray'), type(self).__name__ + " must have attribute 'InitialTensionsArray'. Make sure to run " + type(self).__name__ + ".find_initial_tension(ReturnMultipleInitialTensions=True) first."
		assert i in range(len(self.InitialTensions)), "i must be in range(len(" + type(self).__name__ + ".InitialTensions))."
		self.T_o = self.InitialTensionsArray[i]
	def find_initial_tension_velocity(self,**kwargs):
		"""
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Returns an initial tension velocity of shape (2,).

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) ParticularSolution - Must be a numpy array of shape (2,). Default is numpy.array([0,0]). Must be in the nullspace of [self.m1.R(X_o[0]),self.m2.R(X_o[0])].

		2) Return - must be either true or false (default). When false, function will assign the initial tension value to self.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		"""
		assert hasattr(self,'T_o'), type(self).__name__ + " must have attribute 'T_o'. Make sure to run " + type(self).__name__ + ".find_initial_tension(ReturnMultipleInitialTensions=False) first."

		ParticularSolution = kwargs.get("ParticularSolution",np.array([0,0]))
		assert np.shape(ParticularSolution)==(2,) and str(type(ParticularSolution))=="<class 'numpy.ndarray'>", "ParticularSolution must be a (2,) numpy.ndarray"
		assert (
					abs(
						self.m1.R(self.InitialAngle)*ParticularSolution[0]
						+ self.m2.R(self.InitialAngle)*ParticularSolution[1]
					)
					< 1e-6
				), \
			"ParticularSolution must be in the nullspace of [self.m1.R,self.m2.R]."

		Return = kwargs.get("Return",False)
		assert type(Return)==bool, "Return must be either True or False (default). When Return is False, tension velocity values will be attributed to self."

		HomogeneousSolution = (
			(
				(
					self.Inertia*self.InitialAngularJerk
					+ (
						self.MgL_2
						* self.InitialAngularVelocity
						* np.cos(self.InitialAngle)
					)
					- self.InitialAngularVelocity * (
						self.m1.dR(self.InitialAngle)*self.T_o[0]
						+ self.m2.dR(self.InitialAngle)*self.T_o[1]
					)
				)
				/ (
					self.m1.R(self.InitialAngle)**2
					+ self.m2.R(self.InitialAngle)**2
				)
			)
			* np.array([
				self.m1.R(self.InitialAngle),
				self.m2.R(self.InitialAngle)
			])
		)
		if Return==False:
			self.dT_o = (
				ParticularSolution
				+ HomogeneousSolution
			)
		else:
			return(ParticularSolution+HomogeneousSolution)
	def find_initial_tension_acceleration(self,**kwargs):
		"""
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Returns an initial tension acceleration of shape (2,).

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) ParticularSolution - Must be a numpy array of shape (2,). Default is numpy.array([0,0]). Must be in the nullspace of [self.m1.R(X_o[0]),self.m2.R(X_o[0])].

		2) Return - must be either true or false (default). When false, function will assign the initial tension value to self.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		"""

		assert hasattr(self,'T_o'), type(self).__name__ + " must have attribute 'T_o'. Make sure to run " + type(self).__name__ + ".find_initial_tension(ReturnMultipleInitialTensions=False) first."

		assert hasattr(self,'dT_o'), type(self).__name__ + " must have attribute 'dT_o'. Make sure to run " + type(self).__name__ + ".find_initial_tension_velocity() first."

		ParticularSolution = kwargs.get("ParticularSolution",np.array([0,0]))
		assert np.shape(ParticularSolution)==(2,) and str(type(ParticularSolution))=="<class 'numpy.ndarray'>", "ParticularSolution must be a (2,) numpy.ndarray"
		assert (
					abs(
						self.m1.R(self.InitialAngle)*ParticularSolution[0]
						+ self.m2.R(self.InitialAngle)*ParticularSolution[1]
					)
					< 1e-6
				), \
			"ParticularSolution must be in the nullspace of [self.m1.R,self.m2.R]."

		Return = kwargs.get("Return",False)
		assert type(Return)==bool, "Return must be either True or False (default). When Return is False, tension velocity values will be attributed to self."

		HomogeneousSolution = (
			(
				(
					self.Inertia*self.InitialAngularSnap
					+ self.MgL_2*(
						(
							self.InitialAngularAcceleration
							* np.cos(self.InitialAngle)
						)
						- (
							self.InitialAngularVelocity**2
							* np.sin(self.InitialAngle)
						)
					)
					- self.InitialAngularVelocity**2 * (
						self.m1.d2R(self.InitialAngle)*self.T_o[0]
						+ self.m2.d2R(self.InitialAngle)*self.T_o[1]
					)
					- self.InitialAngularAcceleration * (
						self.m1.dR(self.InitialAngle)*self.T_o[0]
						+ self.m2.dR(self.InitialAngle)*self.T_o[1]
					)
					- 2*self.InitialAngularVelocity * (
						self.m1.dR(self.InitialAngle)*self.dT_o[0]
						+ self.m2.dR(self.InitialAngle)*self.dT_o[1]
					)
				)
				/ (
					self.m1.R(self.InitialAngle)**2
					+ self.m2.R(self.InitialAngle)**2
				)
			)
			* np.array([
				self.m1.R(self.InitialAngle),
				self.m2.R(self.InitialAngle)
			])
		)

		if Return==False:
				self.d2T_o = (
					ParticularSolution
					+ HomogeneousSolution
				)
		else:
			return(ParticularSolution+HomogeneousSolution)
	def plot_initial_tension_values(self,**kwargs):
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

		Seed = kwargs.get("Seed",None)
		assert type(Seed) in [float,int] or Seed is None, "Seed must be a float or an int or None."
		np.random.seed(Seed)

		NumberOfPoints = kwargs.get("NumberOfPoints",1000)
		assert type(NumberOfPoints) == int, "NumberOfPoints must be an int."

		fig = plt.figure(figsize=(10,8))
		ax1 = plt.gca()

		DescriptiveTitle = "Plotting Randomly Generated\nInitial Tendon Tensions"

		ax1.set_title(DescriptiveTitle,Fontsize=20,y=1)

		#Bound Constraints
		ax1.plot(
			[self.TensionBounds[0][0],self.TensionBounds[0][1]],
			[self.TensionBounds[1][0],self.TensionBounds[1][0]],
			'k--'
		)
		ax1.plot(
			[self.TensionBounds[0][0],self.TensionBounds[0][1]],
			[self.TensionBounds[1][1],self.TensionBounds[1][1]],
			'k--'
		)
		ax1.plot(
			[self.TensionBounds[0][0],self.TensionBounds[0][0]],
			[self.TensionBounds[1][0],self.TensionBounds[1][1]],
			'k--'
		)
		ax1.plot(
			[self.TensionBounds[0][1],self.TensionBounds[0][1]],
			[self.TensionBounds[1][0],self.TensionBounds[1][1]],
			'k--'
		)

		Constraint = lambda T1: (
			(
				self.Inertia*self.InitialAngularAcceleration
				+ self.MgL_2*np.sin(self.InitialAngle)
				- self.m1.R(self.InitialAngle)*T1
			)
			/ self.m2.R(self.InitialAngle)
		) # Returns T2 given T1
		Input1 = np.linspace(
			self.TensionBounds[0][0]-0.2*np.diff(self.TensionBounds[0]),
			self.TensionBounds[0][1]+0.2*np.diff(self.TensionBounds[0]),
			1001)
		Input2 = Constraint(Input1)
		ax1.plot(Input1,Input2,'r',lw=2)
		statusbar = dsb(0,NumberOfPoints,title=type(self).__name__+"."+self.plot_initial_tension_values.__name__)
		for i in range(NumberOfPoints):
			T = self.find_initial_tension(
				ReturnMultipleInitialTensions=False,
				Return=True
			)
			ax1.plot(T[0],T[1],'go')
			statusbar.update(i)
		ax1.set_xlabel(r'$T_{1}$',fontsize=14)
		ax1.set_ylabel(r'$T_{2}$',fontsize=14)
		ax1.set_xlim([Input1[0],Input1[-1]])
		ax1.set_ylim([min(Input2),max(Input2)])
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.set_aspect('equal')
		plt.show()
	def initialize_tendon_tension(self,**kwargs):
		"""
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Assigns T_o, dT_o, and d2T_o to class.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) Seed - must be a float or an int. Default is None (seeded by current time).

		2) Bounds - must be a list of size (2,2) or None (in which case the self.TensionBounds will be used)

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		"""
		Seed = kwargs.get("Seed",None)
		Bounds = kwargs.get("Bounds",None)

		self.find_initial_tension(Seed=Seed,Bounds=Bounds)
		self.find_initial_tension_velocity()
		self.find_initial_tension_acceleration()

	def find_initial_muscle_velocity(self):
		"""
		This function will find the initial muscle velocities, given some initial tension and some initial conditions for the plant.
		"""
		assert hasattr(self,"T_o"), \
				type(self).__name__ + " must have attr 'T_o' (Initial Tension). Please run " + type(self).__name__ + ".find_initial_tension(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"dT_o"), \
				type(self).__name__ + " must have attr 'dT_o' (Initial Tension Velocity). Please run " + type(self).__name__ + ".find_initial_tension_velocity(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."

		X_o = [
			self.InitialAngle,
			self.InitialAngularVelocity,
			self.T_o[0],
			self.T_o[1]
		]
		InitialMuscleVelocity1 = (
			(
				self.m1.v_MTU(X_o) * (
					1
					- np.exp(
						-self.T_o[0]
						/ (self.m1.F_MAX*self.m1.cT*self.m1.kT)
					)
				)
				- (
					self.m1.lTo*self.dT_o[0]
					/ (self.m1.F_MAX*self.m1.cT*np.cos(self.m1.pa))
				)
			)
			/ (
				1
				- np.exp(
					-self.T_o[0]
					/ (self.m1.F_MAX*self.m1.cT*self.m1.kT)
				)
			)
		)

		InitialMuscleVelocity2 = (
			(
				self.m2.v_MTU(X_o) * (
					1
					- np.exp(
						-self.T_o[1]
						/ (self.m2.F_MAX*self.m2.cT*self.m2.kT)
					)
				)
				- (
					self.m2.lTo*self.dT_o[1]
					/ (self.m2.F_MAX*self.m2.cT*np.cos(self.m2.pa))
				)
			)
			/ (
				1
				- np.exp(
					-self.T_o[1]
					/ (self.m2.F_MAX*self.m2.cT*self.m2.kT)
				)
			)
		)

		self.Vm_o = np.array([
			InitialMuscleVelocity1,
			InitialMuscleVelocity2
		])
	def find_initial_muscle_acceleration(self):
		"""
		This function will find the initial muscle velocities, given some initial tension and some initial conditions for the plant.
		"""
		assert hasattr(self,"T_o"), \
				type(self).__name__ + " must have attr 'T_o' (Initial Tension). Please run " + type(self).__name__ + ".find_initial_tension(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"dT_o"), \
				type(self).__name__ + " must have attr 'dT_o' (Initial Tension Velocity). Please run " + type(self).__name__ + ".find_initial_tension_velocity(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"d2T_o"), \
				type(self).__name__ + " must have attr 'd2T_o' (Initial Tension Acceleration). Please run " + type(self).__name__ + ".find_initial_tension_acceleration(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."

		assert hasattr(self,"Vm_o"), type(self).__name__ + " must have attr 'Vm_o'. Please run " + type(self).__name__ + ".find_initial_muscle_velocity() before calling this function."

		X_o = [
			self.InitialAngle,
			self.InitialAngularVelocity,
			self.T_o[0],
			self.T_o[1]
		]
		InitialMuscleAcceleration1 = (
			(
				self.T_o[0] * (
					np.exp(-self.T_o[0]/(self.m1.F_MAX*self.m1.cT*self.m1.kT))
					* (
						self.m1.v_MTU(X_o)
						- np.cos(self.m1.pa)*self.Vm_o[0]
					)
					/ (self.m1.lTo*self.m1.kT)
				)
				- self.d2T_o[0]
				+ self.m1.F_MAX*self.m1.cT*self.m1.a_MTU(X_o) * (
					1
					- np.exp(
						-self.T_o[0]
						/ (self.m1.F_MAX*self.m1.cT*self.m1.kT)
					)
				) / self.m1.lTo
			)
			/ (
				self.m1.F_MAX*self.m1.cT * (
					1
					- np.exp(
						-self.T_o[0]
						/ (self.m1.F_MAX*self.m1.cT*self.m1.kT)
					)
				) / (self.m1.lTo*np.cos(self.m1.pa))
			)
		)

		InitialMuscleAcceleration2 = (
			(
				self.dT_o[1] * (
					np.exp(-self.T_o[1]/(self.m2.F_MAX*self.m2.cT*self.m2.kT))
					* (
						self.m2.v_MTU(X_o)
						- np.cos(self.m2.pa)*self.Vm_o[1]
					)
					/ (self.m2.lTo*self.m2.kT)
				)
				- self.d2T_o[1]
				+ self.m2.F_MAX*self.m2.cT*self.m2.a_MTU(X_o) * (
					1
					- np.exp(
						-self.T_o[1]
						/ (self.m2.F_MAX*self.m2.cT*self.m2.kT)
					)
				) / self.m2.lTo
			)
			/ (
				self.m2.F_MAX*self.m2.cT * (
					1
					- np.exp(
						-self.T_o[1]
						/ (self.m2.F_MAX*self.m2.cT*self.m2.kT)
					)
				) / (self.m2.lTo*np.cos(self.m2.pa))
			)
		)

		self.Am_o = np.array([
			InitialMuscleAcceleration1,
			InitialMuscleAcceleration2
		])
	def find_initial_muscle_length_bounds(self,LB_cutoff=0.75,UB_cutoff=1.25):
		"""
		For some given pretensioning on the tendons and some ICs on the plant, this function will return the bounds for initial muscle lengths such that initial activation will be positive.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) LB_cutoff - Must be a number less than 1. Default is 0.75

		1) UB_cutoff - Must be a number greater than 1. Default is 1.25

		"""

		is_number(LB_cutoff,"LB_cutoff",default=0.75,notes="Must be less than 1 and greater than 0.")
		assert 0<LB_cutoff<1, "LB_cutoff must be less than 1, but greater than 0."

		is_number(UB_cutoff,"UB_cutoff",default=1.25,notes="Must be greater than 1.")
		assert UB_cutoff>1, "UB_cutoff must be greater than 1."


		assert hasattr(self,"Vm_o") and hasattr(self,"Am_o"), \
			type(self).__name__ + " must have both attrs 'Vm_o' and 'Am_o'. Please run their respective functions before continuing."

		self.InitialActivation1_func = lambda MuscleLength1: (
			(
				self.T_o[0]*np.cos(self.m1.pa)
				+ (
					self.m1.m
					* self.Vm_o[0]**2
					* np.tan(self.m1.pa)**2
					/ MuscleLength1
				)
				- self.m1.m*self.Am_o[0]
				+ (
					self.m1.bm
					* self.m1.F_MAX
					* np.cos(self.m1.pa)**2
					* self.Vm_o[0]
				)
				- (
					self.m1.F_PE1(MuscleLength1,self.Vm_o[0])
					* self.m1.F_MAX
					* np.cos(self.m1.pa)**2
				)
			)
			/ (
				self.m1.FLV(MuscleLength1,self.Vm_o[0])
				* self.m1.F_MAX
				* np.cos(self.m1.pa)**2
			)
		)
		self.InitialActivation2_func = lambda MuscleLength2: (
			(
				self.T_o[1]*np.cos(self.m2.pa)
				+ (
					self.m2.m
					* self.Vm_o[1]**2
					* np.tan(self.m2.pa)**2
					/ MuscleLength2
				)
				- self.m2.m*self.Am_o[1]
				+ (
					self.m2.bm
					* self.m2.F_MAX
					* np.cos(self.m2.pa)**2
					* self.Vm_o[1]
				)
				- (
					self.m2.F_PE1(MuscleLength2,self.Vm_o[1])
					* self.m2.F_MAX
					* np.cos(self.m2.pa)**2
				)
			)
			/ (
				self.m2.FLV(MuscleLength2,self.Vm_o[1])
				* self.m2.F_MAX
				* np.cos(self.m2.pa)**2
			)
		)

		L1_UB = fsolve(self.InitialActivation1_func,1.5*self.m1.lo)
		L2_UB = fsolve(self.InitialActivation2_func,1.5*self.m2.lo)

		L1_LB = 0.75*self.m1.lo
		if L1_UB > 1.25*self.m1.lo:
			L1_UB = 1.25*self.m1.lo

		L2_LB = 0.75*self.m2.lo
		if L2_UB > 1.25*self.m2.lo:
			L2_UB = 1.25*self.m2.lo

		self.InitialMuscleLengths_Bounds = [[L1_LB,L1_UB],[L2_LB,L2_UB]]

	def return_initial_muscle_lengths_and_activations(self,**kwargs):
		"""
		This function returns initial muscle lengths and muscle activations for a given pretensioning level, as derived from (***insert file_name here for scratchwork***) for a system with reference trajectory pendulum_eqns.reference_trajectories._01.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) PlotBool - Must be either True or False. Default is False. Will plot all possible initial muscle lengths and activations for a given pretensioning level.

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		"""
		PlotBool = kwargs.get("PlotBool",False)
		assert type(PlotBool)==bool,"PlotBool must be a boolean. Default is False."

		assert hasattr(self,"T_o"), \
				type(self).__name__ + " must have attr 'T_o' (Initial Tension). Please run " + type(self).__name__ + ".find_initial_tension(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"dT_o"), \
				type(self).__name__ + " must have attr 'dT_o' (Initial Tension Velocity). Please run " + type(self).__name__ + ".find_initial_tension_velocity(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"d2T_o"), \
				type(self).__name__ + " must have attr 'd2T_o' (Initial Tension Acceleration). Please run " + type(self).__name__ + ".find_initial_tension_acceleration(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."

		X_o = [
			self.InitialAngle,
			self.InitialAngularVelocity,
			self.T_o[0],
			self.T_o[1]
		]
		self.find_initial_muscle_velocity()
		self.find_initial_muscle_acceleration()
		self.find_initial_muscle_length_bounds()

		L1 = np.linspace(
			self.InitialMuscleLengths_Bounds[0][0], self.InitialMuscleLengths_Bounds[0][1],
			1001
		)
		U1 = np.array(list(map(self.InitialActivation1_func,L1)))


		L2 = np.linspace(
			self.InitialMuscleLengths_Bounds[1][0], self.InitialMuscleLengths_Bounds[1][1],
			1001
		)
		U2 = np.array(list(map(self.InitialActivation2_func,L2)))

		if PlotBool == True:
			plt.figure(figsize=(10,8))
			plt.title(r"Viable Initial $l_{m,1}$ and $u_{1}$ Values")
			plt.xlabel(r"$l_{m,1}$ (m)",fontsize=14)
			plt.ylabel(r"$u_{1}$",fontsize=14)
			plt.scatter(L1,U1)
			plt.plot([self.m1.lo,self.m1.lo],[0,1],'0.70',linestyle='--')
			plt.gca().set_ylim((0,1))
			plt.gca().set_xticks(
				[0.25*self.m1.lo,
				0.5*self.m1.lo,
				0.75*self.m1.lo,
				self.m1.lo,
				1.25*self.m1.lo,
				1.5*self.m1.lo,
				1.75*self.m1.lo]
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
			plt.plot([self.m2.lo,self.m2.lo],[0,1],'0.70',linestyle='--')
			plt.gca().set_ylim((0,1))
			plt.gca().set_xticks(
				[0.25*self.m2.lo,
				0.5*self.m2.lo,
				0.75*self.m2.lo,
				self.m2.lo,
				1.25*self.m2.lo,
				1.5*self.m2.lo,
				1.75*self.m2.lo]
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
	def find_initial_activations_given_muscle_lengths_and_tendon_tensions(self,InitialMuscleLengths):
		"""
		This function returns initial muscle activations for a fixed muscle length and a given pretensioning level, as derived from (***insert file_name here for scratchwork***) for the system that starts from rest. (Ex. pendulum_eqns.reference_trajectories._01).

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		NONE

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		"""

		assert hasattr(self,"T_o"), \
				type(self).__name__ + " must have attr 'T_o' (Initial Tension). Please run " + type(self).__name__ + ".find_initial_tension(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"dT_o"), \
				type(self).__name__ + " must have attr 'dT_o' (Initial Tension Velocity). Please run " + type(self).__name__ + ".find_initial_tension_velocity(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."
		assert hasattr(self,"d2T_o"), \
				type(self).__name__ + " must have attr 'd2T_o' (Initial Tension Acceleration). Please run " + type(self).__name__ + ".find_initial_tension_acceleration(**kwargs) before running " + type(self).__name__ + ".return_initial_muscle_lengths_and_activations(**kwargs)."

		assert (len(InitialMuscleLengths)==2
				and type(InitialMuscleLengths)==list), \
			"InitialMuscleLengths must be a list of length 2."

		X_o = [
			self.InitialAngle,
			self.InitialAngularVelocity,
			self.T_o[0],
			self.T_o[1]
		]
		self.find_initial_muscle_velocity()
		self.find_initial_muscle_acceleration()
		self.find_initial_muscle_length_bounds()

		assert (self.InitialMuscleLengths_Bounds[0][0]
				<= InitialMuscleLengths[0]
				<= self.InitialMuscleLengths_Bounds[0][1]), \
			"For the chosen tension level, fixed muscle length produces infeasible activations."
		assert (self.InitialMuscleLengths_Bounds[1][0]
				<= InitialMuscleLengths[1]
				<= self.InitialMuscleLengths_Bounds[1][1]), \
			"For the chosen tension level, fixed muscle length produces infeasible activations."

		InitialActivation1 = self.InitialActivation1_func(
			InitialMuscleLengths[0]
		)
		assert 0<=InitialActivaiton1<=1, "Activaiton1 should be in [0,1)"

		InitialActivation2 = self.InitialActivation2_func(
			InitialMuscleLengths[1]
		)
		assert 0<=InitialActivation2<=1, "Activation2 should be in [0,1)"

		self.Lm_o = InitialMuscleLengths
		self.U_o = np.array([InitialActivation1,InitialActivation2])

	def find_viable_initial_values(self,**kwargs):
		"""
		This function returns initial conditions for the system that follows a reference trajectory. (Ex. pendulum_eqns.reference_trajectories._01)

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		**kwargs

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		1) ReturnAll - Can return all initial values for a given tension level. Will be fed through to return_initial_muscle_lengths_and_activations.

		2) Seed - Can see the random tension generated. When FixedInitialTension is provided, this seed will apply only to the initial conditions for activation and muscle length.

		3) FixedInitialMuscleLengths - must be a list of length 2 or None (Default). If is None, then program will assign this value randomly. Used for trials where we wish to hold muscle length constant for different tension levels.
		"""

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

		if FixedInitialMuscleLengths is None: # Default is None
			L1,U1,L2,U2 = self.return_initial_muscle_lengths_and_activations(
				PlotBool=False
			)
			if ReturnAll==True: # Default False
				return(self.T_o,L1,L2,U1,U2)
			else:
				# PositiveActivations = False
				# while PositiveActivations == False:
				# 	rand_index = np.random.choice(len(L1),2)
				# 	u1,u2 = U1[rand_index[0]],U2[rand_index[1]]
				# 	if u1>0 and u2>0:
				# 		PositiveActivations = True
				# return(
				# 	T,
				# 	np.array([L1[rand_index[0]],L2[rand_index[1]]]),
				# 	np.array([U1[rand_index[0]],U2[rand_index[1]]])
				# 	)
				l1_index = np.where(
					(L1-0.8*self.m1.lo)**2==min((L1-0.8*self.m1.lo)**2)
				)[0][0]
				l2_index = np.where(
					(L2-1.2*self.m2.lo)**2==min((L2-1.2*self.m2.lo)**2)
				)[0][0]
				self.Lm_o = np.array(
					[L1[l1_index],L2[l2_index]]
				)
				self.U_o = np.array([U1[l1_index],U2[l2_index]])
		else:
			self.find_initial_activations_given_muscle_lengths_and_tendon_tensions(FixedInitialMuscleLengths)

			assert ReturnAll != True, "For fixed initial muscle lengths, ReturnAll must be False as the function return_initial_activations_given_muscle_lengths_and_tendon_tensions only returns single values."

	def set_X_o(self,X_o=None):
		if X_o is None:
			self.X[:,0] = [
				self.InitialAngle,
				self.InitialAngularVelocity,
				self.T_o[0],
				self.T_o[1],
				self.Lm_o[0],
				self.Lm_o[1],
				self.Vm_o[0],
				self.Vm_o[1]
			]
		else:
			self.X[:,0] = X_o
	def set_U_o(self,U_o=None):
		if U_o is None:
			self.U[:,0] = self.U_o.T
		else:
			self.U[:,0] = U_o

	def update_MAs(self,x):
		self.R1 = self.m1.R(x[0])
		self.dR1 = self.m1.dR(x[0])
		self.d2R1 = self.m1.d2R(x[0])

		self.R2 = self.m2.R(x[0])
		self.dR2 = self.m2.dR(x[0])
		self.d2R2 = self.m2.d2R(x[0])

	def update_KTs(self,x):
		self.KT1 = self.m1.KT(x[2])
		self.dKT1 = self.m1.dKT(x[2])

		self.KT2 = self.m2.KT(x[3])
		self.dKT2 = self.m2.dKT(x[3])

	def update_F_PE1s(self,x):
		self.F_PE1_1 = self.m1.F_PE1(x[4],x[6])
		self.F_PE1_2 = self.m2.F_PE1(x[5],x[7])

	def update_FLVs(self,x):
		self.FLV1 = self.m1.FLV(x[4],x[6])
		self.FLV2 = self.m2.FLV(x[5],x[7])

	def update_MTUs(self,x):
		self.v_MTU_1 = self.m1.v_MTU(x)
		self.a_MTU_1 = self.m1.a_MTU(x)
		self.v_MTU_2 = self.m2.v_MTU(x)
		self.a_MTU_2 = self.m2.a_MTU(x)

	def dX1_dt(self,x):
		return(x[1])
	def d2X1_dt2(self,x):
		return(
			(
				-self.MgL_2*np.sin(x[0])
				+ self.R1*x[2]
				+ self.R2*x[3]
			) / self.Inertia
		)

	def dX2_dt(self,x):
		return(
			(
				-self.MgL_2*np.sin(x[0])
				+ self.R1*x[2]
				+ self.R2*x[3]
			) / self.Inertia
		)
	def d2X2_dt2(self,x):
		return(
			(
				-self.MgL_2*np.cos(x[0])*x[1]
				+ self.dR1*x[1]*x[2]
				+ self.R1*(self.KT1*(self.v_MTU_1 - self.m1.C1*x[6]))
				+ self.dR2*x[1]*x[3]
				+ self.R2*(self.KT2*(self.v_MTU_2 - self.m2.C1*x[7]))
			) / self.Inertia
		)

	def dX3_dt(self,x):
		return(self.KT1*(self.v_MTU_1 - self.m1.C1*x[6]))

	def dX4_dt(self,x):
		return(self.KT2*(self.v_MTU_2 - self.m2.C1*x[7]))

	def dX5_dt(self,x):
		return(x[6])

	def dX6_dt(self,x):
		return(x[7])

	def dX7_dt(self,x,u):
		return(
			self.m1.C2*x[2]
			- self.m1.C3*self.F_PE1_1
			- self.m1.C4*x[6]
			+ self.m1.C5*x[6]**2/x[4]
			- self.m1.C3*self.FLV1*u[0]
		)

	def dX8_dt(self,x,u):
		return(
			self.m2.C2*x[3]
			- self.m2.C3*self.F_PE1_2
			- self.m2.C4*x[7]
			+ self.m2.C5*x[7]**2/x[5]
			- self.m2.C3*self.FLV2*u[1]
		)
	#################################
	############ IB eqns ############
	#################################

	def update_Z1(self,t,x):
		return(r(t) - x[0])
	def update_dZ1(self,t,x):
		return(dr(t) - self.dX1)
	def update_d2Z1(self,t,x):
		return(d2r(t) - self.dX2)
	def update_d3Z1(self,t,x):
		return(d3r(t) - self.d2X2)

	def update_A1(self,t,x):
		return(dr(t) + k1*self.Z1)
	def update_dA1(self,t,x):
		return(d2r(t) + k1*self.dZ1)
	def update_d2A1(self,t,x):
		return(d3r(t) + k1*self.d2Z1)
	def update_d3A1(self,t,x):
		return(d4r(t) + k1*self.d3Z1)

	def update_Z2(self,t,x):
		return(x[1] - self.A1)
	def update_dZ2(self,t,x):
		"""
		dZ2(t,X,U) = MgL_2*np.sin(X[0]) + Inertia*self.m1.R(X[0])*U[0] + Inertia*self.m2.R(X[0])*U[1] - dA1(t,X)
		"""
		return(self.dX2 - self.dA1)
	def update_d2Z2(self,t,x):
		return(self.d2X2 - self.d2A1)

	def update_A2(self,t,x):
		return(
			self.Z1
			+ self.dA1
			+ self.MgL_2*np.sin(x[0])/self.Inertia
			- k2*self.Z2
		)
	def update_dA2(self,t,x):
		return(
			self.dZ1
			+ self.d2A1
			+ self.MgL_2*np.cos(x[0])*self.dX1/self.Inertia
			- k2*self.dZ2
		)
	def update_d2A2(self,t,x):
		return(
			self.d2Z1
			+ self.d3A1
			- self.MgL_2*np.sin(x[0])*(self.dX1**2)/self.Inertia
			+ self.MgL_2*np.cos(x[0])*self.d2X1/self.Inertia
			- k2*self.d2Z2
		)

	def update_Z3(self,t,x):
		return(
			self.R1*x[2]/self.Inertia
			+ self.R2*x[3]/self.Inertia
			- self.A2
		)
	def update_dZ3(self,t,x):
		"""
		dZ3(t,X) = Inertia*self.m1.dR(X[0])*X[1]*X[2] + Inertia*self.m2.dR(X[0])*X[1]*X[3] \
							+ Inertia*self.m1.R(X[0])*self.m1.KT(X[2])*self.m1.v_MTU(X) - Inertia*self.m1.C1*self.m1.R(X[0])*self.m1.KT(X[2])*U[0] \
								+ Inertia*self.m2.R(X[0])*self.m2.KT(X[3])*self.m2.v_MTU(X) - Inertia*self.m2.C1*self.m2.R(X[0])*self.m2.KT(X[3])*U[1] \
									- dA2(t,X)
		"""
		return(
			(
				self.dR1*x[1]*x[2]
				+ self.dR2*x[1]*x[3]
				+ self.R1*self.KT1*self.v_MTU_1
				- self.m1.C1*self.R1*self.KT1*x[6]
				+ self.R2*self.KT2*self.v_MTU_2
				- self.m2.C1*self.R2*self.KT2*x[7]
				- self.dA2*self.Inertia
			) / self.Inertia
		)

	def update_A3(self,t,x):
		return(
			self.Z2
			- self.dA2
			+ k3*self.Z3
			+ (
				self.dR1*self.dX1*x[2]
				+ self.dR2*self.dX1*x[3]
				+ self.R1*self.KT1*self.v_MTU_1
				+ self.R2*self.KT2*self.v_MTU_2
			) / self.Inertia
		)
	def update_dA3(self,t,x):
		return(
			self.dZ2
			- self.d2A2
			+ k3*self.dZ3
			+ (
				self.d2R1*(self.dX1**2)*x[2]
				+ self.dR1*self.d2X1*x[2]
	 			+ self.dR1*self.dX1*self.dX3
				+ self.d2R2*(self.dX1**2)*x[3]
		 		+ self.dR2*self.d2X1*x[3]
	  			+ self.dR2*self.dX1*self.dX4
				+ self.dR1*self.dX1*self.KT1*self.v_MTU_1
				+ self.R1*self.dKT1*self.dX3*self.v_MTU_1
			 	+ self.R1*self.KT1*self.a_MTU_1
				+ self.dR2*self.dX1*self.KT2*self.v_MTU_2
				+ self.R2*self.dKT2*self.dX4*self.v_MTU_2
				+ self.R2*self.KT2*self.a_MTU_2
			) / self.Inertia
		)
	def update_Z4(self,t,x):
		return(
			(
				self.m1.C1*self.R1*self.KT1*x[6]
				+ self.m2.C1*self.R2*self.KT2*x[7]
				- self.A3*self.Inertia
			) / self.Inertia
		)
	def update_dZ4(self,t,x,u):
		"""
		dZ4 = 	Inertia*self.m1.C1*self.m1.dR(X[0])*dX1_dt(X)*self.m1.KT(X[2])*X[6]\
					+ Inertia*self.m1.C1*self.m1.R(X[0])*self.m1.dKT(X[2])*dX3_dt(X)*X[6]\
						+ Inertia*self.m1.C1*self.m1.R(X[0])*self.m1.KT(X[2])*dX7_dt(X)\
				+ Inertia*self.m2.C1*self.m2.dR(X[0])*dX1_dt(X)*self.m2.KT(X[3])*X[7]\
					+ Inertia*self.m2.C1*self.m2.R(X[0])*self.m2.dKT(X[3])*dX4_dt(X)*X[7]\
						+ Inertia*self.m2.C1*self.m2.R(X[0])*self.m2.KT(X[3])*dX8_dt(X)\
				- dA3(t,X)
		"""
		return(
			(
				self.m1.C1*self.dR1*self.dX1*self.KT1*x[6]
				+ self.m1.C1*self.R1*self.dKT1*self.dX3*x[6]
				+ self.m1.C1*self.R1*self.KT1 * (
					self.m1.C2*x[2]
					- self.m1.C3*self.F_PE1_1
					- self.m1.C4*x[6]
					+ self.m1.C5*x[6]**2/x[4]
				)
				- self.m1.C1*self.m1.C3*self.R1*self.KT1*self.FLV1*u[0]
				+ self.m2.C1*self.dR2*self.dX1*self.KT2*x[7]
			 	+ self.m2.C1*self.R2*self.dKT2*self.dX4*x[7]
				+ self.m2.C1*self.R2*self.KT2 * (
					self.m2.C2*x[3]
					- self.m2.C3*self.F_PE1_2
					- self.m2.C4*x[7]
					+ self.m2.C5*x[7]**2/x[5]
				)
				- self.m2.C1*self.m2.C3*self.R2*self.KT2*self.FLV2*u[1]
				- self.dA3*self.Inertia
			) / self.Inertia
		)
	def update_A4(self,t,x):
		"""
		Inertia*self.m1.C1*self.m1.C3*self.m1.R(X[0])*self.m1.KT(X[2])*self.m1.FLV(X[4],X[6])*U[0] \
			+ Inertia*self.m2.C1*self.m2.C3*self.m2.R(X[0])*self.m2.KT(X[3])*self.m2.FLV(X[5],X[7])*U[1] = \
						Inertia*self.m1.C1*self.m1.dR(X[0])*dX1_dt(X)*self.m1.KT(X[2])*X[6] \
						+ Inertia*self.m1.C1*self.m1.R(X[0])*self.m1.dKT(X[2])*dX3_dt(X)*X[6] \
						+ Inertia*self.m1.C1*self.m1.R(X[0])*self.m1.KT(X[2])*(self.m1.C2*X[2] - self.m1.C3*self.m1.F_PE1(X[4],X[6]) - self.m1.C4*X[6] + self.m1.C5*X[6]**2/X[4]) \
						+ Inertia*self.m2.C1*self.m2.dR(X[0])*dX1_dt(X)*self.m2.KT(X[3])*X[7] \
						+ Inertia*self.m2.C1*self.m2.R(X[0])*self.m2.dKT(X[3])*dX4_dt(X)*X[7] \
						+ Inertia*self.m2.C1*self.m2.R(X[0])*self.m2.KT(X[3])*(self.m2.C2*X[3] - self.m2.C3*self.m2.F_PE1(X[5],X[7]) - self.m2.C4*X[7] + self.m2.C5*X[7]**2/X[5]) \
						- dA3(t,X) - Z3(t,X) + k4*Z4(t,X)
		"""

		return(
			(
				self.m1.C1*self.dR1*self.dX1*self.KT1*x[6]
				+ self.m1.C1*self.R1*self.dKT1*self.dX3*x[6]
				+ self.m1.C1*self.R1*self.KT1*(
					self.m1.C2*x[2]
					- self.m1.C3*self.F_PE1_1
					- self.m1.C4*x[6]
					+ self.m1.C5*x[6]**2/x[4]
				)
				+ self.m2.C1*self.dR2*self.dX1*self.KT2*x[7]
			 	+ self.m2.C1*self.R2*self.dKT2*self.dX4*x[7]
				+ self.m2.C1*self.R2*self.KT2*(
					self.m2.C2*x[3]
					- self.m2.C3*self.F_PE1_2
					- self.m2.C4*x[7]
					+ self.m2.C5*x[7]**2/x[5]
				)
			) / self.Inertia
			- self.dA3
			- self.Z3
			+ k4*self.Z4
		)
	def update_IB_variables(self,t,x):
		self.Z1 = self.update_Z1(t,x)
		self.A1 = self.update_A1(t,x)
		self.Z2 = self.update_Z2(t,x)
		self.dZ1 = self.update_dZ1(t,x)
		self.dA1 = self.update_dA1(t,x)
		self.A2 = self.update_A2(t,x)
		self.Z3 = self.update_Z3(t,x)
		self.dZ2 = self.update_dZ2(t,x)
		self.d2Z1 = self.update_d2Z1(t,x)
		self.d2A1 = self.update_d2A1(t,x)
		self.dA2 = self.update_dA2(t,x)
		self.A3 = self.update_A3(t,x)
		self.Z4 = self.update_Z4(t,x)
		self.dZ3 = self.update_dZ3(t,x)
		self.d3Z1 = self.update_d3Z1(t,x)
		self.d3A1 = self.update_d3A1(t,x)
		self.d2Z2 = self.update_d2Z2(t,x)
		self.d2A2 = self.update_d2A2(t,x)
		self.dZ3 = self.update_dZ3(t,x)
		self.dA3 = self.update_dA3(t,x)

		# self.dZ4 = self.update_dZ4(t,x)
		self.A4 = self.update_A4(t,x)
	def return_constraint_variables(self):
		Coefficient1 = (
			self.m1.C1
			* self.m1.C3
			* self.R1
			* self.KT1
			* self.FLV1
			/ self.Inertia
		)
		Coefficient2 = (
			self.m2.C1
			* self.m2.C3
			* self.R2
			* self.KT2
			* self.FLV2
			/ self.Inertia
		)
		Constraint = self.A4
		return(Coefficient1,Coefficient2,Constraint)
	def return_U_given_sinusoidal_u1(self,i,u1,**kwargs):
	    """
	    Takes in current step (i) and previous input scalar u1 and returns the input U (shape (2,)) for this time step.

	    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    **kwargs
	    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	    1) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Activation_Bounds.

	    """

	    Bounds = kwargs.get("Bounds",[[0,1],[0,1]])
	    assert type(Bounds) == list and np.shape(Bounds) == (2,2), "Bounds for Muscle Activation Control must be a (2,2) list."
	    assert Bounds[0][0]<Bounds[0][1],"Each set of bounds must be in ascending order."
	    assert Bounds[1][0]<Bounds[1][1],"Each set of bounds must be in ascending order."

	    Coefficient1,Coefficient2,Constraint1 = self.return_constraint_variables()
	    assert Coefficient1!=0 and Coefficient2!=0, "Error with Coefficients. Shouldn't both be zero"
	    if Constraint1 < 0:
	    	assert not(Coefficient1 > 0 and Coefficient2 > 0), "Infeasible activations. (Constraint1 < 0, Coefficient1 > 0, Coefficient2 > 0)"
	    if Constraint1 > 0:
	    	assert not(Coefficient1 < 0 and Coefficient2 < 0), "Infeasible activations. (Constraint1 > 0, Coefficient1 < 0, Coefficient2 < 0)"

	    u2 = (Constraint1 - Coefficient1*u1)/Coefficient2
	    NextU = np.array([u1,u2])

	    assert (Bounds[0][0]<=u1<=Bounds[0][1]) and (Bounds[1][0]<=u2<=Bounds[1][1]), "Error! Choice of u1 results in infeasible activation along backstepping constraint."

	    return(NextU)

	def update_pendulum_variables(self,i):
		"""
		Need to find the input for i+1 first.
		"""
		self.update_MAs(self.X[:,i])
		self.update_MTUs(self.X[:,i])
		self.update_FLVs(self.X[:,i])
		self.update_F_PE1s(self.X[:,i])
		self.update_KTs(self.X[:,i])

		self.dX1 = self.dX1_dt(self.X[:,i])
		self.dX2 = self.dX2_dt(self.X[:,i])
		self.dX3 = self.dX3_dt(self.X[:,i])
		self.dX4 = self.dX4_dt(self.X[:,i])

		self.d2X1 = self.d2X1_dt2(self.X[:,i])
		self.d2X2 = self.d2X2_dt2(self.X[:,i])

		self.update_IB_variables(self.Time[i],self.X[:,i])
		self.U[:,i+1] = self.return_U_given_sinusoidal_u1(i,self.U[0,i+1])

		self.dX5 = self.dX5_dt(self.X[:,i])
		self.dX6 = self.dX6_dt(self.X[:,i])
		self.dX7 = self.dX7_dt(self.X[:,i],self.U[:,i+1])
		self.dX8 = self.dX8_dt(self.X[:,i],self.U[:,i+1])

	def forward_integrate(self,i):
		"""
		Need to update the variables first.
		"""
		self.X[0,i+1] = self.X[0,i] + self.dt*self.dX1
		self.X[1,i+1] = self.X[1,i] + self.dt*self.dX2
		self.X[2,i+1] = self.X[2,i] + self.dt*self.dX3
		self.X[3,i+1] = self.X[3,i] + self.dt*self.dX4
		self.X[4,i+1] = self.X[4,i] + self.dt*self.dX5
		self.X[5,i+1] = self.X[5,i] + self.dt*self.dX6
		self.X[6,i+1] = self.X[6,i] + self.dt*self.dX7
		self.X[7,i+1] = self.X[7,i] + self.dt*self.dX8

"""
################################
###### Plotting Functions ######
################################
"""

def plot_states(t,X,**kwargs):
	"""
	Takes in a numpy.ndarray for time (t) of shape (N,) and the numpy.ndarray for the state space (X) of shape (M,N), where M is the number of states and N is the same length as time t. Returns a plot of the states.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	**kwargs
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Return - must be a bool. Determines if the function returns a function handle. Default is False.

	2) InputString - must be a string. Input to the DescriptiveTitle that can be used to personalize the title. Default is None.

	"""
	import numpy as np
	import matplotlib.pyplot as plt

	assert (np.shape(X)[0] in [2,4,8]) \
				and (np.shape(X)[1] == len(t)) \
					and (str(type(X)) == "<class 'numpy.ndarray'>"), \
			"X must be a (2,N), (4,N), or (8,N) numpy.ndarray, where N is the length of t."


	Return = kwargs.get("Return",False)
	assert type(Return)==bool, "Return must be either True or False."

	InputString = kwargs.get("InputString",None)
	assert InputString is None or type(InputString)==str, "InputString must either be None or a str."

	Normalized = kwargs.get("Normalized",False)
	assert type(Normalized)==bool, "Normalized must be either True or False."

	NumStates = np.shape(X)[0]
	X[:2,:] = 180*X[:2,:]/np.pi # converting to deg and deg/s
	NumRows = int(np.ceil(NumStates/4))
	if NumStates < 4:
		NumColumns = NumStates
	else:
		NumColumns = 4

	ColumnNumber = [el%4 for el in np.arange(0,NumStates,1)]
	RowNumber = [int(el/4) for el in np.arange(0,NumStates,1)]
	if Normalized==False:
		Units = ["(Deg)","(Deg/s)","(N)","(N)","(m)","(m)","(m/s)","(m/s)"]
	else:
		Units = ["(Deg)","(Deg/s)","(N)","(N)",r"($\hat{l}_{o}$)",r"($\hat{l}_{o}$)",r"($\hat{l}_{o}/s$)",r"($\hat{l}_{o}/s$)"]
	if InputString is None:
		DescriptiveTitle = "Plotting States vs. Time"
	else:
		assert type(InputString)==str, "InputString must be a string"
		DescriptiveTitle = InputString + " Driven"
	if NumRows == 1:
		FigShape = (NumColumns,)
	else:
		FigShape = (NumRows,NumColumns)
	Figure = kwargs.get("Figure",None)
	assert (Figure is None) or \
				(	(type(Figure)==tuple) and \
					(str(type(Figure[0]))=="<class 'matplotlib.figure.Figure'>") and\
					(np.array([str(type(ax))=="<class 'matplotlib.axes._subplots.AxesSubplot'>" \
						for ax in Figure[1].flatten()]).all()) and \
					(Figure[1].shape == FigShape)\
				),\
				 	("Figure can either be left blank (None) or it must be constructed from data that has the same shape as X.\ntype(Figure) = " + str(type(Figure)) + "\ntype(Figure[0]) = " + str(type(Figure[0])) + "\nFigure[1].shape = " + str(Figure[1].shape) + " instead of (" + str(NumRows) + "," + str(NumColumns) + ")" + "\ntype(Figure[1].flatten()[0]) = " + str(type(Figure[1].flatten()[0])))
	if Figure is None:
		fig, axes = plt.subplots(NumRows,NumColumns,figsize=(3*NumColumns,2*NumRows + 2))
		plt.subplots_adjust(top=0.85,bottom=0.15,left=0.075,right=0.975)
		plt.suptitle(DescriptiveTitle,Fontsize=20,y=0.975)
		if NumStates<=4:
			for j in range(NumStates):
				axes[ColumnNumber[j]].spines['right'].set_visible(False)
				axes[ColumnNumber[j]].spines['top'].set_visible(False)
				axes[ColumnNumber[j]].plot(t,X[j,:])
				if ColumnNumber[j]!=0:
					axes[ColumnNumber[j]].set_xticklabels(\
										[""]*len(axes[ColumnNumber[j]].get_xticks()))
				else:
					axes[ColumnNumber[j]].set_xlabel("Time (s)")
				axes[ColumnNumber[j]].set_title(r"$x_{" + str(j+1) + "}$ " + Units[j])
		else:
			for j in range(NumStates):
				axes[RowNumber[j],ColumnNumber[j]].spines['right'].set_visible(False)
				axes[RowNumber[j],ColumnNumber[j]].spines['top'].set_visible(False)
				if Normalized==False:
					axes[RowNumber[j],ColumnNumber[j]].plot(t,X[j,:])
				elif Normalized==True:
					if j in [4,6]:
						axes[RowNumber[j],ColumnNumber[j]].plot(t,X[j,:]/BIC.lo)
					elif j in [5,7]:
						axes[RowNumber[j],ColumnNumber[j]].plot(t,X[j,:]/TRI.lo)
					else:
						axes[RowNumber[j],ColumnNumber[j]].plot(t,X[j,:])
				if not((RowNumber[j]==RowNumber[-1]) and (ColumnNumber[j]==0)):
					axes[RowNumber[j],ColumnNumber[j]].set_xticklabels(\
										[""]*len(axes[RowNumber[j],ColumnNumber[j]].get_xticks()))
				else:
					axes[RowNumber[j],ColumnNumber[j]].set_xlabel("Time (s)")
				axes[RowNumber[j],ColumnNumber[j]].set_title(r"$x_{" + str(j+1) + "}$ "+ Units[j])
			if NumStates%4!=0:
				[fig.delaxes(axes[RowNumber[-1],el]) for el in range(ColumnNumber[-1]+1,4)]
	else:
		fig = Figure[0]
		axes = Figure[1]
		for i in range(NumStates):
			if NumRows != 1:
				if Normalized==False:
					axes[RowNumber[i],ColumnNumber[i]].plot(t,X[i,:])
				elif Normalized==True:
					if i in [4,6]:
						axes[RowNumber[i],ColumnNumber[i]].plot(t,X[i,:]/BIC.lo)
					elif i in [5,7]:
						axes[RowNumber[i],ColumnNumber[i]].plot(t,X[i,:]/TRI.lo)
					else:
						axes[RowNumber[i],ColumnNumber[i]].plot(t,X[i,:])
			else:
				axes[ColumnNumber[i]].plot(t,X[i,:])
	X[:2,:] = np.pi*X[:2,:]/180
	if Return == True:
		return((fig,axes))
	else:
		plt.show()

def plot_inputs(t,U,**kwargs):
	"""
	Takes in a numpy.ndarray for time (t) of shape (N,) and the numpy.ndarray for the input (U) (NOT NECESSARILY THE SAME LENGTH AS t). Returns a plot of the states.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	**kwargs
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Return - must be a bool. Determines if the function returns a function handle. Default is False.

	2) InputString - must be a string. Input to the DescriptiveTitle that can be used to personalize the title. Default is None.

	"""
	import numpy as np
	import matplotlib.pyplot as plt

	assert (np.shape(U)[0] == 2) \
				and (np.shape(U)[1] == len(t)) \
					and (str(type(U)) == "<class 'numpy.ndarray'>"), \
			"X must be a (2,N) numpy.ndarray, where N is the length of t."

	Return = kwargs.get("Return",False)
	assert type(Return)==bool, "Return must be either True or False."

	InputString = kwargs.get("InputString",None)
	assert InputString is None or type(InputString)==str, "InputString must either be None or a str."

	if InputString is None:
		DescriptiveTitle = "Plotting Inputs vs. Time"
	else:
		assert type(InputString)==str, "InputString must be a string"
		DescriptiveTitle = InputString + " vs. Time"

	Figure = kwargs.get("Figure",None)
	assert (Figure is None) or \
				(	(type(Figure)==tuple) and \
					(str(type(Figure[0]))=="<class 'matplotlib.figure.Figure'>") and\
					(np.array([str(type(ax))=="<class 'matplotlib.axes._subplots.AxesSubplot'>" \
						for ax in Figure[1].flatten()]).all()) and \
					(Figure[1].shape == (2,))\
				),\
				 	"Figure can either be left blank (None) or it must be constructed from data that has the same shape as U."
	if Figure is None:
		fig, axes = plt.subplots(1,2,figsize=(13,5))
		plt.subplots_adjust(top=0.9,hspace=0.4,bottom=0.1,left=0.075,right=0.975)
		plt.suptitle(DescriptiveTitle,Fontsize=20,y=0.975)

		axes[0].plot(t,U[0,:],lw=2)
		axes[0].plot([-1,t[-1]+1],[0,0],'k--',lw=0.5)
		axes[0].set_xlim([t[0],t[-1]])
		axes[0].spines['right'].set_visible(False)
		axes[0].spines['top'].set_visible(False)
		axes[0].set_ylabel(r"$u_1$")
		axes[0].set_xlabel("Time (s)")

		axes[1].plot(t,U[1,:],lw=2)
		axes[1].plot([-1,t[-1]+1],[0,0],'k--',lw=0.5)
		axes[1].set_xlim([t[0],t[-1]])
		axes[1].spines['right'].set_visible(False)
		axes[1].spines['top'].set_visible(False)
		axes[1].set_ylabel(r"$u_2$")
		axes[1].set_xticks(axes[0].get_xticks())
		axes[1].set_xticklabels([""]*len(axes[0].get_xticks()))
	else:
		fig = Figure[0]
		axes = Figure[1]
		axes[0].plot(t,U[0,:],lw=2)
		axes[1].plot(t,U[1,:],lw=2)

	if Return == True:
		return((fig,axes))
	else:
		plt.show()

def plot_l_m_comparison(t,X,**kwargs):

	"""

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	**kwargs
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	1) Return - must be a bool. Determines if the function returns a function handle. Default is False.

	2) InputString - must be a string. Input to the DescriptiveTitle that can be used to personalize the title. Default is None.

	"""
	import numpy as np
	import matplotlib.pyplot as plt

	assert (np.shape(X)[0] >= 2) \
				and (np.shape(X)[1] == len(t)) \
					and (str(type(X)) == "<class 'numpy.ndarray'>"), \
			"X must be a (M,N) numpy.ndarray, where M is greater than or equal to 2 and N is the length of t."

	Return = kwargs.get("Return",False)
	assert type(Return)==bool, "Return must be either True or False."

	Normalized = kwargs.get("Normalized",False)
	assert type(Normalized)==bool, "Normalized must be either True or False."

	InputString = kwargs.get("InputString",None)
	assert InputString is None or type(InputString)==str, "InputString must either be None or a str."
	if InputString is None:
		DescriptiveTitle = "Muscle vs. Musculotendon Lengths"
	else:
		DescriptiveTitle = "Muscle vs. Musculotendon Lengths\n" + InputString + " Driven"

	L_m = kwargs.get("MuscleLengths",None)
	assert (L_m is None) or (str(type(L_m))=="<class 'numpy.ndarray'>" and np.shape(L_m)==(2,len(t))), "L_m must either be a numpy.ndarray of size (2,N) or left as None (Default)."

	V_m = kwargs.get("MuscleVelocities",None)
	assert (V_m is None) or (str(type(V_m))=="<class 'numpy.ndarray'>" and np.shape(V_m)==(2,len(t))), "V_m must either be a numpy.ndarray of size (2,N) or left as None (Default)."

	ReturnError = kwargs.get("ReturnError",False)
	assert type(ReturnError)==bool, "ReturnError must be either True or False."

	assert L_m is not None or V_m is not None, "Error! Need to input some length/velocity measurement for the muscles."

	if L_m is None:
		"""
		This is for the muscle velocity driven controller. These values of initial muscle length are estimates taken to be the optimal muscle lengths. We will need to run some sensitivity analysis to ensure that this does not drastically effect the deviations from the MTU estimate.
		"""
		l_m1 = integrate.cumtrapz(V_m[0,:],t,initial = 0) + np.ones(len(t))*BIC.lo
		l_m2 = integrate.cumtrapz(V_m[1,:],t,initial = 0) + np.ones(len(t))*TRI.lo
	else:
		l_m1 = L_m[0,:]
		l_m2 = L_m[1,:]

	"""
	Note: X must be transposed in order to run through map()
	"""
	Figure = kwargs.get("Figure",None)
	assert (Figure is None) \
				or (
					(type(Figure)==tuple) and \
					(str(type(Figure[0]))=="<class 'matplotlib.figure.Figure'>") and\
					(np.array([str(type(ax))=="<class 'matplotlib.axes._subplots.AxesSubplot'>" \
						for ax in Figure[1].flatten()]).all()) and \
					(Figure[1].shape == (2,2))\
				),\
				 	"Figure can either be left blank (None) or it must be constructed from data that has the same shape as X."

	if Figure is None:
		fig, axes = plt.subplots(2,2,figsize = (14,7))
		plt.suptitle(DescriptiveTitle,Fontsize=20,y=0.975)
		l_m1_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: BIC.v_MTU(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m1[0]
		l_m2_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: TRI.v_MTU(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m2[0]
		if Normalized==False:
			axes[0,0].plot(t,l_m1_by_MTU_approximation, '0.70')
			axes[0,0].plot(t,l_m1)
			axes[0,1].plot(t,l_m1-l_m1_by_MTU_approximation)
			axes[1,0].plot(t,l_m2_by_MTU_approximation, '0.70')
			axes[1,0].plot(t,l_m2)
			axes[1,1].plot(t,l_m2-l_m2_by_MTU_approximation)

			axes[0,0].set_ylabel(r"$l_{m,1}/l_{MTU,1}$ (m)")
			axes[0,1].set_ylabel("Error (m)")
			axes[1,0].set_ylabel(r"$l_{m,2}/l_{MTU,2}$ (m)")
			axes[1,1].set_ylabel("Error (m)")
		else:
			axes[0,0].plot(t,l_m1_by_MTU_approximation/BIC.lo, '0.70')
			axes[0,0].plot(t,l_m1/BIC.lo)
			axes[0,1].plot(t,(l_m1-l_m1_by_MTU_approximation)/BIC.lo)
			axes[1,0].plot(t,l_m2_by_MTU_approximation/TRI.lo, '0.70')
			axes[1,0].plot(t,l_m2/TRI.lo)
			axes[1,1].plot(t,(l_m2-l_m2_by_MTU_approximation)/TRI.lo)

			axes[0,0].set_ylabel(r"$\l_{m,1}/l_{MTU,1}$ (Norm.)")
			axes[0,1].set_ylabel("Error (Norm.)")
			axes[1,0].set_ylabel(r"$l_{m,2}/l_{MTU,2}$ (Norm.)")
			axes[1,1].set_ylabel("Error (Norm.)")

		axes[0,0].set_xlabel("Time (s)")
		axes[0,1].set_xlabel("Time (s)")
		axes[1,0].set_xlabel("Time (s)")
		axes[1,1].set_xlabel("Time (s)")
	else:
		fig = Figure[0]
		axes = Figure[1]
		l_m1_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: BIC.v_MTU(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m1[0]
		l_m2_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: TRI.v_MTU(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m2[0]
		if Normalized==False:
			axes[0,0].plot(t,l_m1_by_MTU_approximation, '0.70')
			axes[0,0].plot(t,l_m1)
			axes[0,1].plot(t,l_m1-l_m1_by_MTU_approximation)
			axes[1,0].plot(t,l_m2_by_MTU_approximation, '0.70')
			axes[1,0].plot(t,l_m2)
			axes[1,1].plot(t,l_m2-l_m2_by_MTU_approximation)
		else:
			axes[0,0].plot(t,l_m1_by_MTU_approximation/BIC.lo, '0.70')
			axes[0,0].plot(t,l_m1/BIC.lo)
			axes[0,1].plot(t,(l_m1-l_m1_by_MTU_approximation)/BIC.lo)
			axes[1,0].plot(t,l_m2_by_MTU_approximation/TRI.lo, '0.70')
			axes[1,0].plot(t,l_m2/TRI.lo)
			axes[1,1].plot(t,(l_m2-l_m2_by_MTU_approximation)/TRI.lo)

	if Return == True:
		if ReturnError == True:
			return(
				(fig,axes),
				[l_m1-l_m1_by_MTU_approximation,l_m2-l_m2_by_MTU_approximation]
				)
		else:
			return((fig,axes))
	else:
		if ReturnError == True:
			plt.show()
			return([l_m1-l_m1_by_MTU_approximation,l_m2-l_m2_by_MTU_approximation])
		else:
			plt.show()
