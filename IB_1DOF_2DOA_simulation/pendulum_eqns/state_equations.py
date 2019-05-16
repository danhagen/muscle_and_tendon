from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.physiology.musclutendon_equations import *
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

g,L = 9.80, 0.45 #m/s², m
# g,L = 0, 0.45 #m/s², m REMOVING GRAVITY
M = 1.6 # kg

c1 = -(3*g)/(2*L)
c2 = 3/(M*L**2)
c3 = np.cos(α1)
c4 = np.cos(α2)
c5 = np.cos(α1)/m1
c6 = F_MAX1*np.cos(α1)**2/m1
c7 = F_MAX1*bm1*np.cos(α1)**2/(m1*lo1)
c8 = np.tan(α1)**2
c9 = np.cos(α2)/m2
c10 = F_MAX2*np.cos(α2)**2/m2
c11 = F_MAX2*bm2*np.cos(α2)**2/(m2*lo2)
c12 = np.tan(α2)**2

def dX1_dt(X):
	return(X[1])
def d2X1_dt2(X):
	return(dX2_dt(X))

def dX2_dt(X,U=None):
	if U is None:
		return(c1*np.sin(X[0]) + c2*R1(X)*X[2] + c2*R2(X)*X[3])
	else:
		return(c1*np.sin(X[0]) + c2*R1(X)*U[0] + c2*R2(X)*U[1])
def d2X2_dt2(X):
	return(c1*np.cos(X[0])*dX1_dt(X) + c2*dR1_dx1(X)*dX1_dt(X)*X[2] + c2*R1(X)*dX3_dt(X)\
			+ c2*dR2_dx1(X)*dX1_dt(X)*X[3] + c2*R2(X)*dX4_dt(X))

v_MTU1 = return_MTU_velocity([dX1_dt,dX2_dt],
								[r1,dr1_dθ,d2r1_dθ2])
a_MTU1 = return_MTU_acceleration([dX1_dt,dX2_dt],
								[r1,dr1_dθ,d2r1_dθ2])

v_MTU2 = return_MTU_velocity([dX1_dt,dX2_dt],
								[r2,dr2_dθ,d2r2_dθ2])
a_MTU2 = return_MTU_acceleration([dX1_dt,dX2_dt],
								[r2,dr2_dθ,d2r2_dθ2])

def dX3_dt(X,U=None):
	if U is None:
		return(KT_1(X)*(v_MTU1(X) - c3*X[6]))
	else:
		return(KT_1(X)*(v_MTU1(X) - c3*U[0]))

def dX4_dt(X,U=None):
	if U is None:
		return(KT_2(X)*(v_MTU2(X) - c4*X[7]))
	else:
		return(KT_2(X)*(v_MTU2(X) - c4*U[1]))

def dX5_dt(X):
	return(X[6])

def dX6_dt(X):
	return(X[7])

def dX7_dt(X,U):
	return(c5*X[2] - c6*F_PE1_1(X) - c7*X[6] + c8*X[6]**2/X[4] - c6*FLV_1(X)*U[0])

def dX8_dt(X,U):
	return(c9*X[3] - c10*F_PE1_2(X) - c11*X[7] + c12*X[7]**2/X[5] - c10*FLV_2(X)*U[1])

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

	NumStates = np.shape(X)[0]
	X[:2,:] = 180*X[:2,:]/np.pi # converting to deg and deg/s
	NumRows = int(np.ceil(NumStates/5))
	if NumStates < 5:
		NumColumns = NumStates
	else:
		NumColumns = 5

	ColumnNumber = [el%5 for el in np.arange(0,NumStates,1)]
	RowNumber = [int(el/5) for el in np.arange(0,NumStates,1)]
	Units = ["(Deg)","(Deg/s)","(N)","(N)","(m)","(m)","(m/s)","(m/s)"]
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
		if NumStates <=5:
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
				axes[RowNumber[j],ColumnNumber[j]].plot(t,X[j,:])
				if not(RowNumber[j] == RowNumber[-1] and ColumnNumber[j]==0):
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
		l_m1 = integrate.cumtrapz(V_m[0,:],t,initial = 0) + np.ones(len(t))*lo1
		l_m2 = integrate.cumtrapz(V_m[1,:],t,initial = 0) + np.ones(len(t))*lo2
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
										np.array(list(map(lambda X: v_MTU1(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m1[0]
		l_m2_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: v_MTU2(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m2[0]

		axes[0,0].plot(t,l_m1_by_MTU_approximation, '0.70')
		axes[0,0].plot(t,l_m1)
		axes[0,0].set_ylabel(r"$l_{m,1}/l_{MTU,1}$ (m)")
		axes[0,0].set_xlabel("Time (s)")

		axes[0,1].plot(t,l_m1-l_m1_by_MTU_approximation)
		axes[0,1].set_ylabel("Error (m)")
		axes[0,1].set_xlabel("Time (s)")

		axes[1,0].plot(t,l_m2_by_MTU_approximation, '0.70')
		axes[1,0].plot(t,l_m2)
		axes[1,0].set_ylabel(r"$l_{m,2}/l_{MTU,2}$ (m)")
		axes[1,0].set_xlabel("Time (s)")

		axes[1,1].plot(t,l_m2-l_m2_by_MTU_approximation)
		axes[1,1].set_ylabel("Error (m)")
		axes[1,1].set_xlabel("Time (s)")
	else:
		fig = Figure[0]
		axes = Figure[1]
		l_m1_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: v_MTU1(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m1[0]
		l_m2_by_MTU_approximation = integrate.cumtrapz(
										np.array(list(map(lambda X: v_MTU2(X),X.T))),\
										t,initial=0
									) \
									+ np.ones(len(t))*l_m2[0]
		axes[0,0].plot(t,l_m1_by_MTU_approximation, '0.70')
		axes[0,0].plot(t,l_m1)

		axes[0,1].plot(t,l_m1-l_m1_by_MTU_approximation)

		axes[1,0].plot(t,l_m2_by_MTU_approximation, '0.70')
		axes[1,0].plot(t,l_m2)

		axes[1,1].plot(t,l_m2-l_m2_by_MTU_approximation)

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
