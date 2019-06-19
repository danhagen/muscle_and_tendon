import numpy as np
from pendulum_eqns.physiology.muscle_settings import *
from pendulum_eqns.physiology.MA_functions import *
from danpy.useful_functions import *

AllMuscleSettings = return_muscle_settings(PreselectedMuscles=[5,6])

R_Transpose, dR_Transpose, d2R_Transpose = \
			return_MA_matrix_functions(AllMuscleSettings,ReturnMatrixFunction=False,θ_PS=np.pi/2)

BIC_Settings = AllMuscleSettings["BIC"]
TRI_Settings = AllMuscleSettings["TRI"]

"""
R_Transpose, dR_Transpose, and d2R_Transpose are of the form (n,m), where n is the number of muscles and m in the number of joints. In order to unpack the two muscles used in this model, we first must get the elbow MA functions R_Transpose[:,1], then change to a 1xn matrix (by the transpose), and then change to an array to reduce the ndmin from 2 to 1.
"""
params = {
	"Muscle 1" : {
		"Settings" : BIC_Settings,
		"MA" : np.array(R_Transpose[:,1].T)[0][0],
		"dMA" : np.array(dR_Transpose[:,1].T)[0][0],
		"d2MA" : np.array(d2R_Transpose[:,1].T)[0][0]
	},
	"Muscle 2" : {
		"Settings" : TRI_Settings,
		"MA" : np.array(R_Transpose[:,1].T)[0][1],
		"dMA" : np.array(dR_Transpose[:,1].T)[0][1],
		"d2MA" : np.array(d2R_Transpose[:,1].T)[0][1]
	},
	"cT" : 27.8,
	"kT" : 0.0047,
	"LrT" : 0.964,
	"beta" : 1.55,
	"omega" : 0.75,
	"rho" : 2.12,
	"V_max" : -9.15,
	"cv0" : -5.78,
	"cv1" : 9.18,
	"av0" : -1.53,
	"av1" : 0,
	"av2" : 0,
	"bv" : 0.69,
	"c_1" : 23.0,
	"k_1" : 0.046,
	"Lr1" : 1.17,
	"eta" : 0.01
}
class Musculotendon:
	def __init__(self,MuscleNumber,**params):
		assert MuscleNumber in [1,2], "MuscleNumber must be either 1 or 2."

		self.cT = params.get("cT",27.8)
		is_number(self.cT,"cT",default=27.8)
		self.kT = params.get("kT",0.0047)
		is_number(self.kT,"kT",default=0.0047)
		self.LrT = params.get("LrT",0.964)
		is_number(self.LrT,"LrT",default=0.964)

		self.beta = params.get("beta",1.55)
		is_number(self.beta,"beta",default=1.55)
		self.omega = params.get("omega",0.75)
		is_number(self.omega,"omega",default=0.75)
		self.rho = params.get("rho",2.12)
		is_number(self.rho,"rho",default=2.12)

		self.V_max = params.get("V_max",-9.15)
		is_number(self.V_max,"V_max",default=-9.15)
		self.cv0 = params.get("cv0",-5.78)
		is_number(self.cv0,"cv0",default=-5.78)
		self.cv1 = params.get("cv1",9.18)
		is_number(self.cv1,"cv1",default=9.18)
		self.av0 = params.get("av0",-1.53)
		is_number(self.av0,"av0",default=-1.53)
		self.av1 = params.get("av1",0)
		is_number(self.av1,"av1",default=0)
		self.av2 = params.get("av2",0)
		is_number(self.av2,"av2",default=0)
		self.bv = params.get("bv",0.69)
		is_number(self.bv,"bv",default=0.69)

		self.c_1 = params.get("c_1",23.0)
		is_number(self.c_1,"c_1",default=23.0)
		self.k_1 = params.get("k_1",0.046)
		is_number(self.k_1,"k_1",default=0.046)
		self.Lr1 = params.get("Lr1",1.17)
		is_number(self.Lr1,"Lr1",default=1.17)
		self.eta = params.get("eta",0.01)
		is_number(self.eta,"eta",default=0.01)

		assert ("Muscle " + str(MuscleNumber)) in params.keys(), "params must have the muscle settings for Muscle " + str(MuscleNumber)
		self.MuscleSettings = params["Muscle " + str(MuscleNumber)]["Settings"]

		self.pa = unit_conversion(
			return_primary_source(
				self.MuscleSettings["Pennation Angle"]
			)
		) # rads
		is_number(self.pa,"pa")

		self.m = unit_conversion(
			return_primary_source(
				self.MuscleSettings["Mass"]
			)
		) # kg
		is_number(self.m,"m")

		self.bm = 0.01 # kg/s
		is_number(self.bm,"bm")

		self.PCSA = unit_conversion(
			return_primary_source(
				self.MuscleSettings["PCSA"]
			)
		)
		is_number(self.PCSA,"PCSA")

		self.F_MAX = unit_conversion(
			return_primary_source(
				self.MuscleSettings["Maximum Isometric Force"]
			)
		)
		is_number(self.F_MAX,"F_MAX")

		self.L_CE_max = 1.2 # These values must be adjusted (SENSITIVITY ANALYSIS NEEDED!)
		is_number(self.L_CE_max,"L_CE_max")

		self.lo = unit_conversion(
			return_primary_source(
				self.MuscleSettings["Optimal Muscle Length"]
			)
		)
		is_number(self.lo,"lo")

		self.lTo = unit_conversion(
			return_primary_source(
				self.MuscleSettings["Optimal Tendon Length"]
			)
		)
		is_number(self.lTo,"lTo")

		self.R = params["Muscle " + str(MuscleNumber)]["MA"]
		self.dR = params["Muscle " + str(MuscleNumber)]["dMA"]
		self.d2R = params["Muscle " + str(MuscleNumber)]["d2MA"]

	def FL(self,l):
		return(
			np.exp(-abs(((l/self.lo)**self.beta-1)/self.omega)**self.rho)
		)

	def FV(self,l,v):
		return(
			np.piecewise(v,[v<=0, v>0],\
				[
					lambda v: (
						(self.V_max - v/self.lo)
						/ (
							self.V_max
							+ (v/self.lo) * (self.cv0 + self.cv1*(l/self.lo))
						)
					),\
					lambda v: (
						(
							self.bv
							- (v/self.lo) * (
								self.av0
								+ self.av1*(l/self.lo)
								+ self.av2*(l/self.lo)**2
							)
						)
						/ (self.bv + (v/self.lo))
					)
				]
			)
		)

	def KT(self,T):
		return(
			(self.F_MAX*self.cT/self.lTo)
			* (1-np.exp(-T/(self.F_MAX*self.cT*self.kT)))
		) # NOT NORMALIZED (in N/m)

	def dKT(self,T):
		return(
			(1/(self.kT*self.lTo))
			* np.exp(-T/(self.F_MAX*self.cT*self.kT))
		) # NOT NORMALIZED (in N/m)

	def FLV(self,l,v):
		return(self.FL(l)*self.FV(l,v))

	def F_PE1(self,l,v):
		return(
			self.c_1 * self.k_1 * np.log(
				np.exp(
					(l/(self.lo*self.L_CE_max) - self.Lr1)/self.k_1
				)
				+ 1
			)
			+ self.eta*(v/self.lo)
		)

	def add_v_MTU(self,state_equations):
		"""
	    MTU velocity equation for a simple hinge joint with dynamics given by a simple pendulum equation,

	               ẍ = F(x,ẋ)
	                   ↓
	               ⎡ x₁ = x ⎤
	               ⎣ x₂ = ẋ ⎦
	                   ↓
	      ⎡   ẋ₁ = x₂ = f₁(x₁,x₂)     ⎤
	      ⎣ ẋ₂ = F(x₁,x₂) = f₂(x₁,x₂) ⎦

	    where f₁ and f₂ are the state_equations.

	    MA_equations should be a list of length 3 and contain the moment arm function as well as first and second derivative w.r.t. joint angle (x₁). Each function should only take in one value (x₁), but the resulting functions will be of all state variables (X) - as long as X has 2 or more states.

	    returns MTU velocity equation that is a functions of state array X (of length >= 2).

	    NOTE: this value is not normalized (given in meters).
	    """
		assert (len(state_equations) == 2
				and type(state_equations) == list), \
			"state_equations must be a list of length 2."
		ẋ1 = state_equations[0]
		ẋ2 = state_equations[1]

		def v_MTU(X):
		    return(
				np.sign(-self.R(X[0]))
				* ẋ1(X)
				* np.sqrt(self.dR(X[0])**2 + self.R(X[0])**2)
			)

		setattr(self,"v_MTU",v_MTU)

	def add_a_MTU(self,state_equations):
	    """
	    MTU acceleration equation for a simple hinge joint with dynamics given by a simple pendulum equation,

	               ẍ = F(x,ẋ)
	                   ↓
	               ⎡ x₁ = x ⎤
	               ⎣ x₂ = ẋ ⎦
	                   ↓
	      ⎡   ẋ₁ = x₂ = f₁(x₁,x₂)     ⎤
	      ⎣ ẋ₂ = F(x₁,x₂) = f₂(x₁,x₂) ⎦

	    where f₁ and f₂ are the state_equations.

	    MA_equations should be a list of length 3 and contain the moment arm function as well as first and second derivative w.r.t. joint angle (x₁). Each function should only take in one value (x₁), but the resulting functions will be of all state variables (X) - as long as X has 2 or more states.

	    returns MTU acceleration equation that is a functions of state array X (of length >= 2).

	    NOTE: this value is not normalized (given in meters).
	    """
	    assert (len(state_equations) == 2
	            and type(state_equations) == list), \
	        "state_equations must be a list of length 2."
	    assert all([str(type(el))=="<class 'function'>"
	            for el in state_equations]), \
	        "All elements in state_equations must be functions."

	    ẋ1 = state_equations[0]
	    ẋ2 = state_equations[1]

	    def a_MTU(X):
	    	return(
	            np.sign(-self.R(X[0]))*(
	                ẋ2(X)
	                * np.sqrt(self.dR(X[0])**2 + self.R(X[0])**2)
	    			+
	                ẋ1(X)**2
	                * self.dR(X[0])
	                * (self.d2R(X[0]) + self.R(X[0]))
	                / np.sqrt(self.dR(X[0])**2 + self.R(X[0])**2)
	                )
	            )

	    setattr(self,"a_MTU",a_MTU)

BIC = Musculotendon(1,**params)
TRI = Musculotendon(1,**params)
TRI.R = params["Muscle " + str(2)]["MA"]
TRI.dR = params["Muscle " + str(2)]["dMA"]
TRI.d2R = params["Muscle " + str(2)]["d2MA"]
