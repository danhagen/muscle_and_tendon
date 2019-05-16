import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.state_equations import *

def return_muscle_length_from_tension(Time,X):
    Tension1 = X[2,:]
    Tension2 = X[3,:]

    T1_o = X[2,0]
    T2_o = X[3,0]

    lm1_o = X[4,0]
    lm2_o = X[5,0]

    MusculotendonLength1 = cumtrapz(
        [v_MTU1(X[:,i]) for i in range(np.shape(X)[1])],
        x=Time,
        initial=0
    ) # integrates v_MTU1 from t=0 to t=i of shape(Time).

    MusculotendonLength2 = cumtrapz(
        [v_MTU2(X[:,i]) for i in range(np.shape(X)[1])],
        x=Time,
        initial=0
    ) # integrates v_MTU2 from t=0 to t=i of shape(Time).

    MuscleLength1 = np.array(
        list(
            map(
                lambda T1,l_MTU1: (
                    lm1_o
                    + (l_MTU1-MusculotendonLength1[0])/np.cos(α1)
                    - kT*lTo1*np.log(
                        (np.exp(T1/(F_MAX1*kT*cT))-1)
                        / (np.exp(T1_o/(F_MAX1*kT*cT))-1)
                    )/np.cos(α1)
                ),
                Tension1,
                MusculotendonLength1
            )
        )
    )
    MuscleLength2 = np.array(
        list(
            map(
                lambda T2,l_MTU2: (
                    lm2_o
                    + (l_MTU2-MusculotendonLength2[0])/np.cos(α2)
                    - kT*lTo2*np.log(
                        (np.exp(T2/(F_MAX2*kT*cT))-1)
                        / (np.exp(T2_o/(F_MAX2*kT*cT))-1)
                    )/np.cos(α2)
                ),
                Tension2,
                MusculotendonLength2
            )
        )
    )
    return(MuscleLength1,MuscleLength2)

def plot_recovered_vs_simulated_muscle_length(Time,TotalX):
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize=[10,7])
    plt.suptitle("Simulated (gray) vs. Recovered Muscle Lengths")
    for i in range(np.shape(TotalX)[0]):
        MuscleLength1_Recovered,MuscleLength2_Recovered = \
            return_muscle_length_from_tension(
                Time,TotalX[i]
        )

        ax1.plot(Time,TotalX[i,4,:],'0.70',lw=3)
        ax1.plot(Time,MuscleLength1_Recovered)
        ax2.plot(Time,TotalX[i,5,:],'0.70',lw=3)
        ax2.plot(Time,MuscleLength2_Recovered)

    ax1.set_title("Muscle 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Muscle Length (m)")

    ax2.set_title("Muscle 2")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Muscle Length (m)")

    return(fig1)
