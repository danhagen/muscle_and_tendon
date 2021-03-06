import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.state_equations import *

def return_tension_from_muscle_length(Time,X):
    MuscleLength1 = X[4,:]
    MuscleLength2 = X[5,:]

    MusculotendonLength1 = cumtrapz(
        [BIC.v_MTU(X[:,i]) for i in range(np.shape(X)[1])],
        x=Time,
        initial=0
    ) # integrates BIC.v_MTU from t=0 to t=i of shape(Time).

    MusculotendonLength2 = cumtrapz(
        [TRI.v_MTU(X[:,i]) for i in range(np.shape(X)[1])],
        x=Time,
        initial=0
    ) # integrates TRI.v_MTU from t=0 to t=i of shape(Time).

    T1_o = X[2,0]
    T2_o = X[3,0]
    T1 = np.array(
        list(
            map(
                lambda lm1,l_MTU1: (
                    (BIC.F_MAX*BIC.kT*BIC.cT) * np.log(
                        (np.exp(T1_o/(BIC.F_MAX*BIC.kT*BIC.cT))-1) * np.exp(
                            (
                                l_MTU1
                                - MusculotendonLength1[0]
                                - np.cos(BIC.pa)*(
                                    lm1
                                    - MuscleLength1[0]
                                )
                            ) / (BIC.kT*BIC.lTo)
                        )
                        + 1
                    )
                ),
                MuscleLength1,
                MusculotendonLength1
            )
        )
    )
    T2 = np.array(
        list(
            map(
                lambda lm2,l_MTU2: (
                    (TRI.F_MAX*TRI.kT*TRI.cT) * np.log(
                        (np.exp(T2_o/(TRI.F_MAX*TRI.kT*TRI.cT))-1) * np.exp(
                            (
                                l_MTU2
                                - MusculotendonLength2[0]
                                - np.cos(TRI.pa)*(
                                    lm2
                                    - MuscleLength2[0]
                                )
                            ) / (TRI.kT*TRI.lTo)
                        )
                        + 1
                    )
                ),
                MuscleLength2,
                MusculotendonLength2
            )
        )
    )
    return(T1,T2)

def plot_recovered_vs_simulated_tension(Time,TotalX):
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize=[10,7])
    plt.suptitle("Simulated (gray) vs. Recovered Tendon Tensions")
    for i in range(np.shape(TotalX)[0]):
        T1_Recovered,T2_Recovered = return_tension_from_muscle_length(
            Time,TotalX[i]
        )

        ax1.plot(Time,TotalX[i,2,:],'0.70',lw=3)
        ax1.plot(Time,T1_Recovered)
        ax2.plot(Time,TotalX[i,3,:],'0.70',lw=3)
        ax2.plot(Time,T2_Recovered)

    ax1.set_title("Muscle 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Tension (N)")

    ax2.set_title("Muscle 2")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tension (N)")

    return(fig1)
