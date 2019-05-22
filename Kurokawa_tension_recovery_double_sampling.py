##################################################
### Replicating the Plots from Kurokawa (2001) ###
##################################################

import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import dsb
from kurokawa_2001_data import *

Time = np.linspace(-400,0,len(MuscleLength))

params = {
    "kT" : 0.0047,
    "cT" : 27.8,
    "F_MAX" : 3000,
    "lTo" : 0.45,
    "To" : Kurokawa_Tension[0],
    "lo" : 0.05,
    "β" : 1.55,
    "ω" : 0.75,
    "ρ" : 2.12,
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
    "η" : 0.01,
    "L_CE_max" : 1.2,
    "bm" : 0.01,
    "m" : 0.5,
    "bu" : 20
}

def FL(l,**params):
    lo = params.get("lo",0.06)
    β = params.get("β",1.55)
    ω = params.get("ω",0.75)
    ρ = params.get("ρ",2.12)
    return(
        np.exp(-abs(((l/lo)**β-1)/ω)**ρ)
    )
def FV(l,v,**params):
    V_max = params.get("V_max", -9.15)
    cv0 = params.get("cv0", -5.78)
    cv1 = params.get("cv1", 9.18)
    av0 = params.get("av0", -1.53)
    av1 = params.get("av1", 0)
    av2 = params.get("av2", 0)
    bv = params.get("bv", 0.69)
    lo = params.get("lo",0.06)
    if v<=0:
        return((V_max - v/lo)/(V_max + (cv0 + cv1*(l/lo))*(v/lo)))
    else:
        return((bv-(av0 + av1*(l/lo) + av2*(l/lo)**2)*(v/lo))/(bv + (v/lo)))
def FLV(l,v,**params):
	return(FL(l,**params)*FV(l,v,**params))
def F_PE1(l,v,**params):
    c_1 = params.get("c_1", 23.0)
    k_1 = params.get("k_1", 0.046)
    lo = params.get("lo", 0.06)
    L_CE_max = params.get("L_CE_max", 1.2)
    Lr1 = params.get("Lr1", 1.17)
    η = params.get("η", 0.01)
    return(c_1*k_1*np.log(np.exp((l/(lo*L_CE_max) - Lr1)/k_1) + 1) + η*(v/lo))
def return_tension_from_muscle_length(
        MuscleLength,
        MusculotendonLength,
        Pennation=None,
        **params
        ):

    if Pennation is None:
        Pennation = [0]*len(MuscleLength)
    elif type(Pennation)!=list:
        Pennation = [Pennation]*len(MuscleLength)

    kT = params.get("kT", 0.0047)
    cT = params.get("cT", 27.8)
    F_MAX = params.get("F_MAX", 1000)
    lTo = params.get("lTo", 0.45)
    T_i = params.get("To", 80)
    lm_i = MuscleLength[0]
    l_MTU_i = MusculotendonLength[0]
    α_i = Pennation[0]

    Recovered_Tension = np.array(
        list(
            map(
                lambda lm,l_MTU,α: (
                    F_MAX*kT*cT*np.log(
                        (np.exp(T_i/(F_MAX*kT*cT))-1)*np.exp(
                            (
                                l_MTU
                                - l_MTU_i
                                - np.cos(α)*lm
                                + np.cos(α_i)*lm_i
                            ) / (kT*lTo)
                        )
                        + 1
                    )
                ),
                MuscleLength,
                MusculotendonLength,
                Pennation
            )
        )
    )
    return(Recovered_Tension)
def return_muscle_activation_from_tension_and_muscle_length(
        Time,
        Tension,
        MuscleLength,
        Pennation,
        **params
        ):
    F_MAX = params.get("F_MAX",1000)
    m = params.get("m",0.5)
    bm = params.get("bm",0.01)
    bu = params.get("bu",1)
    dt = (Time[1]-Time[0])/1000
    MuscleVelocity = np.gradient(MuscleLength,dt)
    MuscleAcceleration = np.gradient(MuscleVelocity,dt)

    Activation = np.array(
        list(
            map(
                lambda l,v,a,p,T: (
                    (
                        T*np.cos(p)
                        -m*(a - v**2*np.tan(p)/l)
                        -F_MAX*(np.cos(p)**2)*(
                            F_PE1(l,v,**params)
                            + bm*v
                        )
                    )
                    / (
                        F_MAX
                        * (np.cos(p)**2)
                        * FLV(l,v,**params)
                    )
                ),
                MuscleLength,
                MuscleVelocity,
                MuscleAcceleration,
                Pennation,
                Tension
            )
        )
    )
    # Gamma = lambda l,v,a,p,T: (
    #     (
    #         T*np.cos(p)
    #         -m*(a - v**2*np.tan(p)/l)
    #         -F_MAX*(np.cos(p)**2)*(
    #             F_PE1(l,v,**params)
    #             + bm*v
    #         )
    #     )
    #     / (
    #         F_MAX
    #         * (np.cos(p)**2)
    #         * FLV(l,v,**params)
    #     )
    # )
    # Tau = lambda l,v,p: (
    #     bu
    #     / (
    #         F_MAX
    #         * (np.cos(p)**2)
    #         * FLV(l,v,**params)
    #     )
    # )
    # FE_Activation = np.zeros(len(MuscleLength))
    # FE_Activation[0] = EMG[0]
    # for i in range(len(MuscleLength)-1):
    #     FE_Activation[i+1] = (
    #         FE_Activation[i]
    #         + (dt / Tau(MuscleLength[i],MuscleVelocity[i],Pennation[i]))
    #         * (
    #             Gamma(
    #                 MuscleLength[i],
    #                 MuscleVelocity[i],
    #                 MuscleAcceleration[i],
    #                 Pennation[i],
    #                 Tension[i]
    #             )
    #             - FE_Activation[i]
    #         )
    #     )
    return(Activation)

N_param = 100
# F_MAX_array = np.linspace(1000,4000,N_param)
F_MAX_array = np.array([3000])
# lTo_array = np.linspace(0.2,0.7,N_param)
lTo_array = np.array([0.50])
cT_array = np.linspace(1,50,N_param)
# cT_array = np.array([27.8])
kT_array = np.linspace(0.001,0.5,N_param)
# kT_array = np.array([0.0047])
Error = np.zeros((len(F_MAX_array),len(lTo_array),len(cT_array),len(kT_array)))
statusbar=dsb(
    0,
    len(F_MAX_array)*len(lTo_array)*len(cT_array)*len(kT_array),
    title="Sweeping Tendon Parameters"
)
for h in range(len(F_MAX_array)):
    for i in range(len(lTo_array)):
        for j in range(len(cT_array)):
            for k in range(len(kT_array)):
                params['F_MAX']=F_MAX_array[h]
                params['lTo']=lTo_array[i]
                params['cT']=cT_array[j]
                params['kT']=kT_array[k]
                Recovered_Tension = return_tension_from_muscle_length(
                    MuscleLength,
                    MusculotendonLength,
                    Pennation=Pennation,
                    **params
                )
                Error[h,i,j,k] = ((Kurokawa_Tension-Recovered_Tension.T)**2).mean()
                statusbar.update(
                    (len(kT_array)*len(cT_array)*len(lTo_array))*h
                    + (len(kT_array)*len(cT_array))*i
                    + len(kT_array)*j
                    + k
                )

best_F_MAX = F_MAX_array[np.where(Error==Error.min())[0][0]]
best_lTo = lTo_array[np.where(Error==Error.min())[1][0]]
best_cT = cT_array[np.where(Error==Error.min())[2][0]]
best_kT = kT_array[np.where(Error==Error.min())[3][0]]

params["F_MAX"]=best_F_MAX
params["lTo"]=best_lTo
params["cT"]=best_cT
params["kT"]=best_kT

Recovered_Tension = return_tension_from_muscle_length(
    MuscleLength,
    MusculotendonLength,
    Pennation=Pennation,
    **params
)
delay = 2
lo_array = np.linspace(0.02,0.2,100)
# lo_array = np.array([0.06])
# bm_array = np.linspace(0.01,1,100)
bm_array = np.array([0.01])
L_CE_max_array = np.linspace(0.5,1.5,100)
# L_CE_max_array = np.array([1.2])
# η_array = np.linspace(0.01,1.5,100)
η_array = np.array([0.01])
EMG_Error = np.zeros(
    (len(lo_array),len(bm_array),len(L_CE_max_array),len(η_array))
)
statusbar = dsb(
    0,
    len(lo_array)*len(bm_array)*len(L_CE_max_array)*len(η_array),
    title="Sweeping Muscle Parameters"
)
for i in range(len(lo_array)):
    for j in range(len(bm_array)):
        for k in range(len(L_CE_max_array)):
            for l in range(len(η_array)):
                params['lo']=lo_array[i]
                params['bm']=bm_array[j]
                params['L_CE_max']=L_CE_max_array[k]
                params['η']=η_array[l]
                Recovered_Activation = return_muscle_activation_from_tension_and_muscle_length(
                    Time,
                    Recovered_Tension,
                    MuscleLength,
                    Pennation,
                    **params
                )
                Adjusted_Activation = (
                    max(EMG)
                    * Recovered_Activation
                    / (Recovered_Activation.max())
                )
                EMG_Error[i,j,k] = ((EMG[:len(EMG)-delay]-Adjusted_Activation[delay:])**2).mean()
                statusbar.update(
                    (len(bm_array)*len(L_CE_max_array)*len(η_array))*i
                    + (len(L_CE_max_array)*len(η_array))*j
                    + len(η_array)*k
                    + l
                )

best_lo = lo_array[np.where(EMG_Error==EMG_Error.min())[0][0]]
best_bm = bm_array[np.where(EMG_Error==EMG_Error.min())[1][0]]
best_L_CE_max = L_CE_max_array[np.where(EMG_Error==EMG_Error.min())[2][0]]
best_η = η_array[np.where(EMG_Error==EMG_Error.min())[3][0]]
params["lo"]=best_lo
params["bm"]=best_bm
params["L_CE_max"]=best_L_CE_max
params["η"]=best_η

Recovered_Activation = return_muscle_activation_from_tension_and_muscle_length(
    Time,
    Recovered_Tension,
    MuscleLength,
    Pennation,
    **params
)

fig, (ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
ax1.plot(Time,Kurokawa_Tension,'0.70',lw=3)
ax1.plot(Time,Recovered_Tension,'b')
ax1.set_title("Recovered vs. Experimental Tension")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Tension (N)")
ax1.legend(["Kurokawa (2001)","Recovered"])

ax2.plot(Time,Kurokawa_Tension-Recovered_Tension,'b')
ax2.set_title("Error")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Error (N)")
ax2.legend([r"$k^T =$ " + "%.3f" % best_kT + " \n " + r"$c^T =$ " + "%.2f" % best_cT + " \n " + r"$l_{T,o} =$ " + "%.3f" % best_lTo + " \n " + r"$F_{MAX} =$ " + "%.1f" % best_F_MAX])

delay = 2 # timesteps
Adjusted_Activation = (
    max(EMG)
    * Recovered_Activation
    / (Recovered_Activation.max())
)
fig3, (ax4,ax5) = plt.subplots(2,1,figsize=[7,10])
ax4.plot(Time,EMG,'0.70',lw=3)
ax4.plot(Time,Recovered_Activation,'r')
ax4.plot(Time[:-delay],Recovered_Activation[delay:],'r--')
ax4.plot(Time,Adjusted_Activation,'b')
ax4.plot(Time[:-delay],Adjusted_Activation[delay:],'b--')
ax4.set_title("Recovered Activation vs. Experimental EMG")
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Activation")
ax4.legend(["Kurokawa (2001)","Recovered","Recovered (" + str(delay*12.5) + "ms Delay)","Adjusted","Adjusted (" + str(delay*12.5) + "ms Delay)"])

ax5.plot(Time,EMG-Adjusted_Activation,'b')
ax5.plot(Time[:-delay],EMG[:-delay]-(Adjusted_Activation)[delay:],'b--')
ax5.set_title("Error")
ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Error (Unitless)")
ax5.legend([r"$l_o =$ " + "%.3f" % params["lo"] + " \n " + r"$b_m =$ " + "%.3f" % params["bm"] + " \n " + r"$L_{CE}^{MAX} =$ " + "%.3f" % params["L_CE_max"], "Delayed by " + str(delay*12.5) + "ms"])

plt.show()
