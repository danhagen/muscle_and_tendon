##################################################
### Replicating the Plots from Kurokawa (2001) ###
##################################################

import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import dsb

Time = np.linspace(-400,0,17)

MuscleLength = [
    56.04752451422728/1000,
    56.02899596943993/1000,
    55.95909282319677/1000,
    55.815075497804244/1000,
    55.62473681044336/1000,
    55.11941286169765/1000,
    54.32521205558563/1000,
    53.04421584551525/1000,
    51.303374842086264/1000,
    48.605787162365395/1000,
    45.606689526559585/1000,
    41.66516272634302/1000,
    38.70396438669314/1000,
    37.17199061541238/1000,
    36.572339529567465/1000,
    36.38705408169404/1000,
    36.248089995788966/1000
] # in m

MusculotendonLength = [
    423.53794408240486/1000,
    423.5694765608577/1000,
    423.5526592390162/1000,
    423.98570527643477/1000,
    423.98570527643477/1000,
    424.2169434517553/1000,
    424.2631910868194/1000,
    424.4502837923061/1000,
    424.85179735127184/1000,
    425.0683203699811/1000,
    425.19024595333195/1000,
    425.3437040151356/1000,
    425.14399831826785/1000,
    424.0319529114989/1000,
    420.03153247845285/1000,
    411.93399201177215/1000,
    401.7384906453647/1000
] # in m

Pennation = [
    0.34600525076608996,
    0.34629398658050276,
    0.3471601940237412,
    0.3500606765230701,
    0.3539586100176432,
    0.3577252999602105,
    0.3640906122324934,
    0.380876662534039,
    0.4034505534790417,
    0.4339778036755978,
    0.4808580004520802,
    0.5284600367645947,
    0.5804062346485023,
    0.6123771639171225,
    0.6266958354409585,
    0.6158419936900763,
    0.613243371360361
] # in radians

Kurokawa_Tension = [
    62.96220633299285,
    61.34831460674158,
    57.9979570990807,
    55.15832482124618,
    54.259448416751795,
    58.161389172625135,
    65.78140960163434,
    84.65781409601635,
    121.06230847803883,
    175.6894790602656,
    261.0418794688458,
    361.06230847803886,
    430.56179775280907,
    441.9203268641471,
    361.06230847803886,
    185.14811031664968,
    17.732379979570993
] # in N

EMG = [
    0.011480568075516626,
    0.019219321370864868,
    0.02729823964622842,
    0.05646738668254103,
    0.06386597499787397,
    0.12143889786546476,
    0.18198826430818948,
    0.2982396462284208,
    0.44587124755506424,
    0.6543923803044477,
    0.7496385747087337,
    0.7725146696147632,
    0.7212347988774556,
    0.6042180457521898,
    0.31337698783910195,
    0.07194489327323751,
    0.07237009949825665
] # Normalized to maximum EMG during the squat-jump

params = {
    "kT" : 0.0047,
    "cT" : 27.8,
    "F_MAX" : 3000,
    "lTo" : 0.45,
    "To" : 80,
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
    "m" : 0.5
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

    MuscleVelocity = np.gradient(MuscleLength,Time[1]-Time[0])
    MuscleAcceleration = np.gradient(MuscleVelocity,Time[1]-Time[0])

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
    return(Activation)

# cT_array = np.linspace(1,100,1000)
# Error = np.zeros(np.shape(cT_array))
# statusbar = dsb(0,len(cT_array),title="Sweeping cT")
# for i in range(len(cT_array)):
#     params['cT']=cT_array[i]
#     Recovered_Tension = return_tension_from_muscle_length(
#         MuscleLength,
#         MusculotendonLength,
#         Pennation=Pennation,
#         **params
#     )
#     Error[i] = ((Kurokawa_Tension-Recovered_Tension.T)**2).mean()
#     statusbar.update(i)
# best_cT = cT_array[np.where(Error==min(Error))]
# params["cT"]=best_cT
#
# lTo_array = np.linspace(0.2,0.7,1000)
# Error = np.zeros(np.shape(lTo_array))
# statusbar = dsb(0,len(lTo_array),title="Sweeping lTo")
# for i in range(len(lTo_array)):
#     params['lTo']=lTo_array[i]
#     Recovered_Tension = return_tension_from_muscle_length(
#         MuscleLength,
#         MusculotendonLength,
#         Pennation=Pennation,
#         **params
#     )
#     Error[i] = ((Kurokawa_Tension-Recovered_Tension.T)**2).mean()
#     statusbar.update(i)
# best_lTo = lTo_array[np.where(Error==min(Error))]
# params["lTo"]=best_lTo

kT_array = np.linspace(0.001,2,1000)
Error = np.zeros(np.shape(kT_array))
statusbar = dsb(0,len(kT_array),title="Sweeping kT")
for i in range(len(kT_array)):
    params['kT']=kT_array[i]
    Recovered_Tension = return_tension_from_muscle_length(
        MuscleLength,
        MusculotendonLength,
        Pennation=Pennation,
        **params
    )
    Error[i] = ((Kurokawa_Tension-Recovered_Tension.T)**2).mean()
    statusbar.update(i)
best_kT = kT_array[np.where(Error==min(Error))]
params["kT"]=best_kT

# F_MAX_array = np.linspace(100,2500,1000)
# Error = np.zeros(np.shape(F_MAX_array))
# statusbar = dsb(0,len(F_MAX_array),title="Sweeping F_MAX")
# for i in range(len(F_MAX_array)):
#     params['F_MAX']=F_MAX_array[i]
#     Recovered_Tension = return_tension_from_muscle_length(
#         MuscleLength,
#         MusculotendonLength,
#         Pennation=Pennation,
#         **params
#     )
#     Error[i] = ((Kurokawa_Tension-Recovered_Tension.T)**2).mean()
#     statusbar.update(i)
# best_F_MAX = F_MAX_array[np.where(Error==min(Error))]
# params["F_MAX"]=best_F_MAX

Recovered_Tension = return_tension_from_muscle_length(
    MuscleLength,
    MusculotendonLength,
    Pennation=Pennation,
    **params
)

Recovered_Activation = return_muscle_activation_from_tension_and_muscle_length(
    Time,
    Recovered_Tension,
    MuscleLength,
    Pennation,
    **params
)

fig, (ax1,ax2) = plt.subplots(2,1,figsize=[7,10])
ax1.plot(Time,Kurokawa_Tension,'0.70',marker="o",lw=3)
ax1.plot(Time,Recovered_Tension,'b',marker="o")
ax1.set_title("Recovered vs. Experimental Tension")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Tension (N)")
ax1.legend(["Kurokawa (2001)","Recovered"])

ax2.plot(Time,Kurokawa_Tension-Recovered_Tension.T[0],'b',marker="o")
ax2.set_title("Error")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Error (N)")
ax2.legend([r"$k^T =$ " + "%.4f" % best_kT[0]])

# fig2 = plt.figure()
# ax3 = plt.gca()
# ax3.plot(kT_array,Error,'b')
# ax3.set_xlabel(r"$k^{T}$")
# ax3.set_ylabel("Mean Squared Error (N)")

delay = 2 # timesteps
Adjusted_Activation = (
    max(EMG)
    * Recovered_Activation
    / (Recovered_Activation.max())
)
fig3, (ax4,ax5) = plt.subplots(2,1,figsize=[7,10])
ax4.plot(Time,EMG,'0.70',marker="o",lw=3)
ax4.plot(Time,Adjusted_Activation,'b',marker="o")
ax4.plot(Time[:-delay],Adjusted_Activation[delay:],'b--',marker='o')
ax4.set_title("Recovered Activation vs. Experimental EMG")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Activation")
ax4.legend(["Kurokawa (2001)","Recovered","Recovered (" + str(delay*25) + "ms Delay)"])

ax5.plot(Time,EMG-Adjusted_Activation.T[0],'b',marker="o")
ax5.plot(Time[:-delay],EMG[:-delay]-(Adjusted_Activation.T[0])[delay:],'b--',marker="o")
ax5.set_title("Error")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Error (Unitless)")
ax5.legend([r"$l_o =$ " + "%.4f" % params["lo"], "Delayed by " + str(delay*25) + "ms"])

plt.show()
