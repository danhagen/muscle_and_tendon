import numpy as np
from pendulum_eqns.physiology.muscle_settings import *
from pendulum_eqns.physiology.MA_functions import *

AllMuscleSettings = return_muscle_settings(PreselectedMuscles=[5,6])

R_Transpose, dR_Transpose, d2R_Transpose = \
			return_MA_matrix_functions(AllMuscleSettings,ReturnMatrixFunction=False,θ_PS=np.pi/2)


"""
R_Transpose, dR_Transpose, and d2R_Transpose are of the form (n,m), where n is the number of muscles and m in the number of joints. In order to unpack the two muscles used in this model, we first must get the elbow MA functions R_Transpose[:,1], then change to a 1xn matrix (by the transpose), and then change to an array to reduce the ndmin from 2 to 1.
"""

cT = 27.8
kT = 0.0047
LrT = 0.964

β = 1.55
ω = 0.75
ρ = 2.12

V_max = -9.15
cv0 = -5.78
cv1 = 9.18
av0 = -1.53
av1 = 0
av2 = 0
bv = 0.69

FL = lambda l,lo: np.exp(-abs(((l/lo)**β-1)/ω)**ρ)
FV = lambda l,v,lo: np.piecewise(v,[v<=0, v>0],\
	[lambda v: (V_max - v/lo)/(V_max + (cv0 + cv1*(l/lo))*(v/lo)),\
	lambda v: (bv-(av0 + av1*(l/lo) + av2*(l/lo)**2)*(v/lo))/(bv + (v/lo))])

c_1 = 23.0
k_1 = 0.046
Lr1 = 1.17
η = 0.01

##########################################
############## BIC SETTINGS ##############
##########################################

BIC_Settings = AllMuscleSettings["BIC"]

α1 = unit_conversion(
	return_primary_source(
		BIC_Settings["Pennation Angle"])) # rads
m1 = unit_conversion(
	return_primary_source(
		BIC_Settings["Mass"])) # kg

bm1 = 0.01 # kg/s

PCSA1 = unit_conversion(
	return_primary_source(
		BIC_Settings["PCSA"]))
F_MAX1 = unit_conversion(
	return_primary_source(
		BIC_Settings["Maximum Isometric Force"]))
L_CE_max_1 = 1.2 # These values must be adjusted (SENSITIVITY ANALYSIS NEEDED!)

lo1 = unit_conversion(
	return_primary_source(
		BIC_Settings["Optimal Muscle Length"]))
lTo1 = unit_conversion(
	return_primary_source(
		BIC_Settings["Optimal Tendon Length"]))

r1 = np.array(R_Transpose[:,1].T)[0][0]
dr1_dθ = np.array(dR_Transpose[:,1].T)[0][0]
d2r1_dθ2 = np.array(d2R_Transpose[:,1].T)[0][0]

def R1(X):
	return(r1(X[0])) #
def dR1_dx1(X):
	return(dr1_dθ(X[0]))
def d2R1_dx12(X):
	return(d2r1_dθ2(X[0]))
def KT_1(X):
	return((F_MAX1*cT/lTo1)*(1-np.exp(-X[2]/(F_MAX1*cT*kT)))) # NOT NORMALIZED (in N/m)
def dKT_1_dx3(X):
	return((1/(kT*lTo1))*np.exp(-X[2]/(F_MAX1*cT*kT))) # NOT NORMALIZED (in N/m)
def FLV_1(X):
	return(FL(X[4],lo1)*FV(X[4],X[6],lo1))
def F_PE1_1(X):
	return(c_1*k_1*np.log(np.exp((X[4]/(lo1*L_CE_max_1) - Lr1)/k_1) + 1) + η*(X[6]/lo1))

##########################################
############## TRI SETTINGS ##############
##########################################

TRI_Settings = AllMuscleSettings["TRI"]

α2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Pennation Angle"])) # rads
m2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Mass"])) # kg

bm2 = 0.01 # kg/s

PCSA2 = unit_conversion(
	return_primary_source(
		TRI_Settings["PCSA"]))
F_MAX2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Maximum Isometric Force"]))
L_CE_max_2 = 1.2 # These values must be adjusted (SENSITIVITY ANALYSIS NEEDED!)

lo2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Optimal Muscle Length"]))
lTo2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Optimal Tendon Length"]))

r2 = np.array(R_Transpose[:,1].T)[0][1]
dr2_dθ = np.array(dR_Transpose[:,1].T)[0][1]
d2r2_dθ2 = np.array(d2R_Transpose[:,1].T)[0][1]

def R2(X):
	return(r2(X[0]))
def dR2_dx1(X):
	return(dr2_dθ(X[0]))
def d2R2_dx12(X):
	return(d2r2_dθ2(X[0]))
def KT_2(X):
	return((F_MAX2*cT/lTo2)*(1-np.exp(-X[3]/(F_MAX2*cT*kT)))) # NOT NORMALIZED (in N/m)
def dKT_2_dx4(X):
	return((1/(kT*lTo2))*np.exp(-X[3]/(F_MAX2*cT*kT))) # NOT NORMALIZED (in N/m)
def FLV_2(X):
	return(FL(X[5],lo2)*FV(X[5],X[7],lo2))
def F_PE1_2(X):
	return(c_1*k_1*np.log(np.exp((X[5]/(lo2*L_CE_max_2) - Lr1)/k_1) + 1) + η*(X[7]/lo2))

##########################################
##########################################
##########################################
