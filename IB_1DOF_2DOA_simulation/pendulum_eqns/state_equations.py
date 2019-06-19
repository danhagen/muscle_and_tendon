from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.reference_trajectories._01 import *
import numpy as np
from scipy import integrate

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
\dot{x}_3 &= K_{T,1}(v_{MTU,1} - c_{3}u_1) \\
\dot{x}_4 &= K_{T,2}(v_{MTU,2} - c_{4}u_2) \\
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

A = -(3*params["g"])/(2*params["L"])
B = 3/(params["M"]*params["L"]**2)

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
	return(A*np.sin(X[0]) + B*BIC.R(X[0])*X[2] + B*TRI.R(X[0])*X[3])

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
	def __init__(self,BIC,TRI,X_o,U_o,Time):
		self.X = np.zeros((8,len(Time)))
		self.X[:,0] = X_o

		self.U = np.zeros((2,len(Time)))
		self.U[:,0] = U_o

		self.Time = Time
		self.dt = Time[1]-Time[0]

	def update_MAs(self,x):
		self.R1 = BIC.R(x[0])
		self.dR1 = BIC.dR(x[0])
		self.d2R1 = BIC.d2R(x[0])

		self.R2 = TRI.R(x[0])
		self.dR2 = TRI.dR(x[0])
		self.d2R2 = TRI.d2R(x[0])

	def update_KTs(self,x):
		self.KT1 = BIC.KT(x[2])
		self.dKT1 = BIC.dKT(x[2])

		self.KT2 = TRI.KT(x[3])
		self.dKT2 = TRI.dKT(x[3])

	def update_F_PE1s(self,x):
		self.F_PE1_1 = BIC.F_PE1(x[4],x[6])
		self.F_PE1_2 = TRI.F_PE1(x[5],x[7])

	def update_FLVs(self,x):
		self.FLV1 = BIC.FLV(x[4],x[6])
		self.FLV2 = TRI.FLV(x[5],x[7])

	def update_MTUs(self,x):
		self.v_MTU_1 = BIC.v_MTU(x)
		self.a_MTU_1 = BIC.a_MTU(x)
		self.v_MTU_2 = TRI.v_MTU(x)
		self.a_MTU_2 = TRI.a_MTU(x)

	def dX1_dt(self,x):
		return(x[1])
	def d2X1_dt2(self,x):
		return(A*np.sin(x[0]) + B*self.R1*x[2] + B*self.R2*x[3])

	def dX2_dt(self,x):
		return(A*np.sin(x[0]) + B*self.R1*x[2] + B*self.R2*x[3])
	def d2X2_dt2(self,x):
		return(
			A*np.cos(x[0])*x[1]
			+ B*self.dR1*x[1]*x[2]
			+ B*self.R1*(self.KT1*(self.v_MTU_1 - BIC.C1*x[6]))
			+ B*self.dR2*x[1]*x[3]
			+ B*self.R2*(self.KT2*(self.v_MTU_2 - TRI.C1*x[7]))
		)

	def dX3_dt(self,x):
		return(self.KT1*(self.v_MTU_1 - BIC.C1*x[6]))

	def dX4_dt(self,x):
		return(self.KT2*(self.v_MTU_2 - TRI.C1*x[7]))

	def dX5_dt(self,x):
		return(x[6])

	def dX6_dt(self,x):
		return(x[7])

	def dX7_dt(self,x,u):
		return(
			BIC.C2*x[2]
			- BIC.C3*self.F_PE1_1
			- BIC.C4*x[6]
			+ BIC.C5*x[6]**2/x[4]
			- BIC.C3*self.FLV1*u[0]
		)

	def dX8_dt(self,x,u):
		return(
			TRI.C2*x[3]
			- TRI.C3*self.F_PE1_2
			- TRI.C4*x[7]
			+ TRI.C5*x[7]**2/x[5]
			- TRI.C3*self.FLV2*u[1]
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
		dZ2(t,X,U) = A*np.sin(X[0]) + B*BIC.R(X[0])*U[0] + B*TRI.R(X[0])*U[1] - dA1(t,X)
		"""
		return(self.dX2 - self.dA1)
	def update_d2Z2(self,t,x):
		return(self.d2X2 - self.d2A1)

	def update_A2(self,t,x):
		return(self.Z1 + self.dA1 - A*np.sin(x[0]) - k2*self.Z2)
	def update_dA2(self,t,x):
		return(
			self.dZ1
			+ self.d2A1
			- A*np.cos(x[0])*self.dX1
			- k2*self.dZ2
		)
	def update_d2A2(self,t,x):
		return(
			self.d2Z1
			+ self.d3A1
			+ A*np.sin(x[0])*(self.dX1**2)
			- A*np.cos(x[0])*self.d2X1
			- k2*self.d2Z2
		)

	def update_Z3(self,t,x):
		return(B*self.R1*x[2] + B*self.R2*x[3] - self.A2)
	def update_dZ3(self,t,x):
		"""
		dZ3(t,X) = B*BIC.dR(X[0])*X[1]*X[2] + B*TRI.dR(X[0])*X[1]*X[3] \
							+ B*BIC.R(X[0])*BIC.KT(X[2])*BIC.v_MTU(X) - B*BIC.C1*BIC.R(X[0])*BIC.KT(X[2])*U[0] \
								+ B*TRI.R(X[0])*TRI.KT(X[3])*TRI.v_MTU(X) - B*TRI.C1*TRI.R(X[0])*TRI.KT(X[3])*U[1] \
									- dA2(t,X)
		"""
		return(
			B*self.dR1*x[1]*x[2]
			+ B*self.dR2*x[1]*x[3]
			+ B*self.R1*self.KT1*self.v_MTU_1
			- B*BIC.C1*self.R1*self.KT1*x[6]
			+ B*self.R2*self.KT2*self.v_MTU_2
			- B*TRI.C1*self.R2*self.KT2*x[7]
			- self.dA2
		)

	def update_A3(self,t,x):
		return(
			self.Z2
			- self.dA2
			+ k3*self.Z3
			+ B*self.dR1*self.dX1*x[2]
			+ B*self.dR2*self.dX1*x[3]
			+ B*self.R1*self.KT1*self.v_MTU_1
			+ B*self.R2*self.KT2*self.v_MTU_2
		)
	def update_dA3(self,t,x):
		return(
			self.dZ2
			- self.d2A2
			+ k3*self.dZ3
			+ B*self.d2R1*(self.dX1**2)*x[2]
			+ B*self.dR1*self.d2X1*x[2]
 			+ B*self.dR1*self.dX1*self.dX3
			+ B*self.d2R2*(self.dX1**2)*x[3]
	 		+ B*self.dR2*self.d2X1*x[3]
  			+ B*self.dR2*self.dX1*self.dX4
			+ B*self.dR1*self.dX1*self.KT1*self.v_MTU_1
			+ B*self.R1*self.dKT1*self.dX3*self.v_MTU_1
		 	+ B*self.R1*self.KT1*self.a_MTU_1
			+ B*self.dR2*self.dX1*self.KT2*self.v_MTU_2
			+ B*self.R2*self.dKT2*self.dX4*self.v_MTU_2
			+ B*self.R2*self.KT2*self.a_MTU_2
		)
	def update_Z4(self,t,x):
		return(
			B*BIC.C1*self.R1*self.KT1*x[6]
			+ B*TRI.C1*self.R2*self.KT2*x[7]
			- self.A3
		)
	def update_dZ4(self,t,x,u):
		"""
		dZ4 = 	B*BIC.C1*BIC.dR(X[0])*dX1_dt(X)*BIC.KT(X[2])*X[6]\
					+ B*BIC.C1*BIC.R(X[0])*BIC.dKT(X[2])*dX3_dt(X)*X[6]\
						+ B*BIC.C1*BIC.R(X[0])*BIC.KT(X[2])*dX7_dt(X)\
				+ B*TRI.C1*TRI.dR(X[0])*dX1_dt(X)*TRI.KT(X[3])*X[7]\
					+ B*TRI.C1*TRI.R(X[0])*TRI.dKT(X[3])*dX4_dt(X)*X[7]\
						+ B*TRI.C1*TRI.R(X[0])*TRI.KT(X[3])*dX8_dt(X)\
				- dA3(t,X)
		"""
		return(
			B*BIC.C1*self.dR1*self.dX1*self.KT1*x[6]
			+ B*BIC.C1*self.R1*self.dKT1*self.dX3*x[6]
			+ B*BIC.C1*self.R1*self.KT1 * (
				BIC.C2*x[2]
				- BIC.C3*self.F_PE1_1
				- BIC.C4*x[6]
				+ BIC.C5*x[6]**2/x[4]
			)
			- B*BIC.C1*BIC.C3*self.R1*self.KT1*self.FLV1*u[0]
			+ B*TRI.C1*self.dR2*self.dX1*self.KT2*x[7]
		 	+ B*TRI.C1*self.R2*self.dKT2*self.dX4*x[7]
			+ B*TRI.C1*self.R2*self.KT2 * (
				TRI.C2*x[3]
				- TRI.C3*self.F_PE1_2
				- TRI.C4*x[7]
				+ TRI.C5*x[7]**2/x[5]
			)
			- B*TRI.C1*TRI.C3*self.R2*self.KT2*self.FLV2*u[1]
			- self.dA3
		)
	def update_A4(self,t,x):
		"""
		B*BIC.C1*BIC.C3*BIC.R(X[0])*BIC.KT(X[2])*BIC.FLV(X[4],X[6])*U[0] \
			+ B*TRI.C1*TRI.C3*TRI.R(X[0])*TRI.KT(X[3])*TRI.FLV(X[5],X[7])*U[1] = \
						B*BIC.C1*BIC.dR(X[0])*dX1_dt(X)*BIC.KT(X[2])*X[6] \
						+ B*BIC.C1*BIC.R(X[0])*BIC.dKT(X[2])*dX3_dt(X)*X[6] \
						+ B*BIC.C1*BIC.R(X[0])*BIC.KT(X[2])*(BIC.C2*X[2] - BIC.C3*BIC.F_PE1(X[4],X[6]) - BIC.C4*X[6] + BIC.C5*X[6]**2/X[4]) \
						+ B*TRI.C1*TRI.dR(X[0])*dX1_dt(X)*TRI.KT(X[3])*X[7] \
						+ B*TRI.C1*TRI.R(X[0])*TRI.dKT(X[3])*dX4_dt(X)*X[7] \
						+ B*TRI.C1*TRI.R(X[0])*TRI.KT(X[3])*(TRI.C2*X[3] - TRI.C3*TRI.F_PE1(X[5],X[7]) - TRI.C4*X[7] + TRI.C5*X[7]**2/X[5]) \
						- dA3(t,X) - Z3(t,X) + k4*Z4(t,X)
		"""

		return(
			B*BIC.C1*self.dR1*self.dX1*self.KT1*x[6]
			+ B*BIC.C1*self.R1*self.dKT1*self.dX3*x[6]
			+ B*BIC.C1*self.R1*self.KT1*(
				BIC.C2*x[2]
				- BIC.C3*self.F_PE1_1
				- BIC.C4*x[6]
				+ BIC.C5*x[6]**2/x[4]
			)
			+ B*TRI.C1*self.dR2*self.dX1*self.KT2*x[7]
		 	+ B*TRI.C1*self.R2*self.dKT2*self.dX4*x[7]
			+ B*TRI.C1*self.R2*self.KT2*(
				TRI.C2*x[3]
				- TRI.C3*self.F_PE1_2
				- TRI.C4*x[7]
				+ TRI.C5*x[7]**2/x[5]
			)
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
		Coefficient1 = B*BIC.C1*BIC.C3*self.R1*self.KT1*self.FLV1
		Coefficient2 = B*TRI.C1*TRI.C3*self.R2*self.KT2*self.FLV2
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
	NumRows = int(np.ceil(NumStates/5))
	if NumStates < 5:
		NumColumns = NumStates
	else:
		NumColumns = 5

	ColumnNumber = [el%5 for el in np.arange(0,NumStates,1)]
	RowNumber = [int(el/5) for el in np.arange(0,NumStates,1)]
	if Normalized==False:
		Units = ["(Deg)","(Deg/s)","(N)","(N)","(m)","(m)","(m/s)","(m/s)"]
	else:
		Units = ["(Deg)","(Deg/s)","(N)","(N)",r"$\hat{l}_{o}$",r"$\hat{l}_{o}$",r"$\hat{l}_{o}/s$",r"$\hat{l}_{o}/s$"]
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
			if NumStates%5!=0:
				[fig.delaxes(axes[RowNumber[-1],el]) for el in range(ColumnNumber[-1]+1,5)]
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
