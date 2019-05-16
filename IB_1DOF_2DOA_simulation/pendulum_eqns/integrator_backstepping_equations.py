from pendulum_eqns.reference_trajectories._01 import *
from pendulum_eqns.state_equations import *

k1,k2,k3,k4 = 100,100,100,10

def Z1(t,X):
	return(r(t) - X[0])
def dZ1(t,X):
	return(dr(t) - dX1_dt(X))
def d2Z1(t,X):
	return(d2r(t) - dX2_dt(X))
def d3Z1(t,X):
	return(d3r(t) - d2X2_dt2(X))
def A1(t,X):
	return(dr(t) + k1*Z1(t,X))
def dA1(t,X):
	return(d2r(t) + k1*dZ1(t,X))
def d2A1(t,X):
	return(d3r(t) + k1*d2Z1(t,X))
def d3A1(t,X):
	return(d4r(t) + k1*d3Z1(t,X))
def Z2(t,X):
	return(X[1] - A1(t,X))
def dZ2(t,X):
	"""
	dZ2(t,X,U) = c1*np.sin(X[0]) + c2*R1(X)*U[0] + c2*R2(X)*U[1] - dA1(t,X)
	"""
	return(dX2_dt(X) - dA1(t,X))
def d2Z2(t,X):
	return(d2X2_dt2(X) - d2A1(t,X))
def A2(t,X):
	return(Z1(t,X) + dA1(t,X) - c1*np.sin(X[0]) - k2*Z2(t,X))
def dA2(t,X):
	return(dZ1(t,X) + d2A1(t,X) - c1*np.cos(X[0])*dX1_dt(X) - k2*dZ2(t,X))
def d2A2(t,X):
	return(d2Z1(t,X) + d3A1(t,X) + c1*np.sin(X[0])*(dX1_dt(X)**2) - c1*np.cos(X[0])*d2X1_dt2(X) - k2*d2Z2(t,X))
def Z3(t,X):
	global r_1,r_2
	return(c2*r_1*X[2] + c2*r_2*X[3] - A2(t,X))
def dZ3(t,X):
	global r_1,kt_1
	global r_2,kt_2
	global dr1_dx1
	global dr2_dx1
	global v_MTU_1,v_MTU_2
	"""
	dZ3(t,X) = c2*dR1_dx1(X)*X[1]*X[2] + c2*dR2_dx1(X)*X[1]*X[3] \
						+ c2*R1(X)*KT_1(X)*v_MTU1(X) - c2*c3*R1(X)*KT_1(X)*U[0] \
							+ c2*R2(X)*KT_2(X)*v_MTU2(X) - c2*c4*R2(X)*KT_2(X)*U[1] \
								- dA2(t,X)
	"""
	return(c2*dr1_dx1*X[1]*X[2] + c2*dr2_dx1*X[1]*X[3] \
					+ c2*r_1*kt_1*v_MTU_1 - c2*c3*r_1*kt_1*X[6] \
						+ c2*r_2*kt_2*v_MTU_2 - c2*c4*r_2*kt_2*X[7] \
							- dA2(t,X))
def A3(t,X,**kwargs):
	InitGlobals = kwargs.get("InitGlobals",False)
	assert type(InitGlobals)==bool, "InitGlobals must be either True or False."

	if InitGlobals == True:
		init_globals_muscle_velocity_controller(X)

	global r_1,kt_1
	global r_2,kt_2
	global dr1_dx1
	global dr2_dx1
	global v_MTU_1,v_MTU_2
	return(Z2(t,X) - dA2(t,X) + k3*Z3(t,X) \
		+ c2*dr1_dx1*dX1_dt(X)*X[2] + 	c2*dr2_dx1*dX1_dt(X)*X[3] \
			+ c2*r_1*kt_1*v_MTU_1 + c2*r_2*kt_2*v_MTU_2)
def dA3(t,X):
	global r_1,kt_1
	global r_2,kt_2
	global dr1_dx1,dkt_1_dx3
	global dr2_dx1,dkt_2_dx4
	global d2r1_dx12,d2r2_dx12
	d2r1_dx12 = d2R1_dx12(X)
	d2r2_dx12 = d2R2_dx12(X)
	global v_MTU_1,a_MTU_1
	v_MTU_1 = v_MTU1(X)
	a_MTU_1 = a_MTU1(X)
	global v_MTU_2,a_MTU_2
	v_MTU_2 = v_MTU2(X)
	a_MTU_2 = a_MTU2(X)

	return(dZ2(t,X) - d2A2(t,X) + k3*dZ3(t,X) \
		+ c2*d2r1_dx12*(dX1_dt(X)**2)*X[2] \
			+ c2*dr1_dx1*d2X1_dt2(X)*X[2] \
	 			+ c2*dr1_dx1*dX1_dt(X)*dX3_dt(X)\
		 + c2*d2r2_dx12*(dX1_dt(X)**2)*X[3] \
	 		+ c2*dr2_dx1*d2X1_dt2(X)*X[3] \
	  			+ c2*dr2_dx1*dX1_dt(X)*dX4_dt(X) \
		+ c2*dr1_dx1*dX1_dt(X)*kt_1*v_MTU_1 \
			+ c2*r_1*dkt_1_dx3*dX3_dt(X)*v_MTU_1 \
			 	+ c2*r_1*kt_1*a_MTU_1 \
		+ c2*dr2_dx1*dX1_dt(X)*kt_2*v_MTU_2 \
			+ c2*r_2*dkt_2_dx4*dX4_dt(X)*v_MTU_2 \
				+ c2*r_2*kt_2*a_MTU_2)
def Z4(t,X):
	global r_1,kt_1
	global r_2,kt_2
	return(c2*c3*r_1*kt_1*X[6] + c2*c4*r_2*kt_2*X[7] - A3(t,X))
def dZ4(t,X,U):
	"""
	dZ4 = 	c2*c3*dR1_dx1(X)*dX1_dt(X)*KT_1(X)*X[6]\
				+ c2*c3*R1(X)*dKT_1_dx3(X)*dX3_dt(X)*X[6]\
					+ c2*c3*R1(X)*KT_1(X)*dX7_dt(X)\
			+ c2*c4*dR2_dx1(X)*dX1_dt(X)*KT_2(X)*X[7]\
				+ c2*c4*R2(X)*dKT_2_dx4(X)*dX4_dt(X)*X[7]\
					+ c2*c4*R2(X)*KT_2(X)*dX8_dt(X)\
			- dA3(t,X)
	"""
	return(	c2*c3*dR1_dx1(X)*dX1_dt(X)*KT_1(X)*X[6] \
				+ c2*c3*R1(X)*dKT_1_dx3(X)*dX3_dt(X)*X[6] \
					+ c2*c3*R1(X)*KT_1(X)*(c5*X[2] - c6*F_PE1_1(X) - c7*X[6] + c8*X[6]**2/X[4]) \
					 	- c2*c3*c6*R1(X)*KT_1(X)*FLV_1(X)*U[0] \
			+ c2*c4*dR2_dx1(X)*dX1_dt(X)*KT_2(X)*X[7] \
			 	+ c2*c4*R2(X)*dKT_2_dx4(X)*dX4_dt(X)*X[7] \
					+ c2*c4*R2(X)*KT_2(X)*(c9*X[3] - c10*F_PE1_2(X) - c11*X[7] + c12*X[7]**2/X[5]) \
					 	- c2*c4*c10*R2(X)*KT_2(X)*FLV_2(X)*U[1] \
			- dA3(t,X))
def A4(t,X,**kwargs):
	"""
	c2*c3*c6*R1(X)*KT_1(X)*FLV_1(X)*U[0] \
		+ c2*c4*c10*R2(X)*KT_2(X)*FLV_2(X)*U[1] = \
					c2*c3*dR1_dx1(X)*dX1_dt(X)*KT_1(X)*X[6] \
					+ c2*c3*R1(X)*dKT_1_dx3(X)*dX3_dt(X)*X[6] \
					+ c2*c3*R1(X)*KT_1(X)*(c5*X[2] - c6*F_PE1_1(X) - c7*X[6] + c8*X[6]**2/X[4]) \
					+ c2*c4*dR2_dx1(X)*dX1_dt(X)*KT_2(X)*X[7] \
					+ c2*c4*R2(X)*dKT_2_dx4(X)*dX4_dt(X)*X[7] \
					+ c2*c4*R2(X)*KT_2(X)*(c9*X[3] - c10*F_PE1_2(X) - c11*X[7] + c12*X[7]**2/X[5]) \
					- dA3(t,X) - Z3(t,X) + k4*Z4(t,X)
	"""
	InitGlobals = kwargs.get("InitGlobals",False)
	assert type(InitGlobals)==bool,"InitGlobals must be either True or False."

	if InitGlobals == True:
		init_globals_muscle_activation_controller(X)

	global r_1,kt_1,flv_1
	global r_2,kt_2,flv_2
	global dr1_dx1,dkt_1_dx3
	global dr2_dx1,dkt_2_dx4
	global f_pe1_1,f_pe1_2

	return(c2*c3*dr1_dx1*dX1_dt(X)*kt_1*X[6] \
				+ c2*c3*r_1*dkt_1_dx3*dX3_dt(X)*X[6] \
					+ c2*c3*r_1*kt_1*(c5*X[2] - c6*f_pe1_1 - c7*X[6] + c8*X[6]**2/X[4]) \
			+ c2*c4*dr2_dx1*dX1_dt(X)*kt_2*X[7] \
			 	+ c2*c4*r_2*dkt_2_dx4*dX4_dt(X)*X[7] \
					+ c2*c4*r_2*kt_2*(c9*X[3] - c10*f_pe1_2 - c11*X[7] + c12*X[7]**2/X[5]) \
			- dA3(t,X) - Z3(t,X) + k4*Z4(t,X))

def init_globals_muscle_activation_controller(X):
	global r_1,kt_1,flv_1
	r_1 = R1(X)
	kt_1 = KT_1(X)
	flv_1 = FLV_1(X)
	global r_2,kt_2,flv_2
	r_2 = R2(X)
	kt_2 = KT_2(X)
	flv_2 = FLV_2(X)
	global dr1_dx1,dkt_1_dx3
	dr1_dx1 = dR1_dx1(X)
	dkt_1_dx3 = dKT_1_dx3(X)
	global dr2_dx1,dkt_2_dx4
	dr2_dx1 = dR2_dx1(X)
	dkt_2_dx4 = dKT_2_dx4(X)
	global f_pe1_1,f_pe1_2
	f_pe1_1 = F_PE1_1(X)
	f_pe1_2 = F_PE1_2(X)

def init_globals_muscle_velocity_controller(X):
	global r_1,kt_1
	r_1 = R1(X)
	kt_1 = KT_1(X)
	global r_2,kt_2
	r_2 = R2(X)
	kt_2 = KT_2(X)
	global dr1_dx1
	dr1_dx1 = dR1_dx1(X)
	global dr2_dx1
	dr2_dx1 = dR2_dx1(X)
	global v_MTU_1,v_MTU_2
	v_MTU_1 = v_MTU1(X)
	v_MTU_2 = v_MTU2(X)
