import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import dsb

Time = np.linspace(-400,0,17)

MuscleLength = [
    56.04752451422728,
    56.02899596943993,
    55.95909282319677,
    55.815075497804244,
    55.62473681044336,
    55.11941286169765,
    54.32521205558563,
    53.04421584551525,
    51.303374842086264,
    48.605787162365395,
    45.606689526559585,
    41.66516272634302,
    38.70396438669314,
    37.17199061541238,
    36.572339529567465,
    36.38705408169404,
    36.248089995788966
] # in mm

MusculotendonLength = [
    423.53794408240486,
    423.5694765608577,
    423.5526592390162,
    423.98570527643477,
    423.98570527643477,
    424.2169434517553,
    424.2631910868194,
    424.4502837923061,
    424.85179735127184,
    425.0683203699811,
    425.19024595333195,
    425.3437040151356,
    425.14399831826785,
    424.0319529114989,
    420.03153247845285,
    411.93399201177215,
    401.7384906453647
] # in mm

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

params = {
    "kT" : 0.0047,
    "cT" : 27.8,
    "F_MAX" : 2000,
    "lTo" : 0.37,
    "To" : 80
}

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

    kT = params.get("kT", 60)
    cT = params.get("cT", 27.8)
    F_MAX = params.get("F_MAX", 1000)
    lTo = params.get("lTo", 0.37)
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

kT_array = np.linspace(1,200,1000)
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

Recovered_Tension = return_tension_from_muscle_length(
    MuscleLength,
    MusculotendonLength,
    Pennation=Pennation,
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

# fig2 = plt.figure()
# ax3 = plt.gca()
# ax3.plot(kT_array,Error,'b')
# ax3.set_xlabel(r"$k^{T}$")
# ax3.set_ylabel("Mean Squared Error (N)")

plt.show()
