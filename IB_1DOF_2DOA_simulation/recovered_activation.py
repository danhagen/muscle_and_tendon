import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from pendulum_eqns.physiology.muscle_params_BIC_TRI import *
from pendulum_eqns.state_equations import *
from recovered_muscle_length import *

def return_muscle_activation_from_tension_and_muscle_length(Time,X):
    Tension1 = X[2,:]
    Tension2 = X[3,:]

    MuscleLength1,MuscleLength2 = return_muscle_length_from_tension(Time,X)
    MuscleVelocity1 = np.gradient(MuscleLength1,Time[1]-Time[0])
    MuscleVelocity2 = np.gradient(MuscleLength2,Time[1]-Time[0])
    MuscleAcceleration1 = np.gradient(MuscleVelocity1,Time[1]-Time[0])
    MuscleAcceleration2 = np.gradient(MuscleVelocity2,Time[1]-Time[0])

    Activation1 = np.array(
        list(
            map(
                lambda l,v,a,T: (
                    (
                        T*np.cos(BIC.pa)
                        -BIC.m*(a - v**2*np.tan(BIC.pa)**2/l)
                        -BIC.F_MAX*np.cos(BIC.pa)**2*(
                            BIC.F_PE1(l,v)
                            + BIC.bm*v
                        )
                    )
                    / (
                        BIC.F_MAX
                        * (np.cos(BIC.pa)**2)
                        * BIC.FLV(l,v)
                    )
                ),
                MuscleLength1,
                MuscleVelocity1,
                MuscleAcceleration1,
                Tension1
            )
        )
    )

    Activation2 = np.array(
        list(
            map(
                lambda l,v,a,T: (
                    (
                        T*np.cos(TRI.pa)
                        -TRI.m*(a - v**2*np.tan(TRI.pa)**2/l)
                        -TRI.F_MAX*np.cos(TRI.pa)**2*(
                            TRI.F_PE1(l,v)
                            + TRI.bm*v
                        )
                    )
                    / (
                        TRI.F_MAX
                        * (np.cos(TRI.pa)**2)
                        * TRI.FLV(l,v)
                    )
                ),
                MuscleLength2,
                MuscleVelocity2,
                MuscleAcceleration2,
                Tension2
            )
        )
    )
    return(Activation1,Activation2)

def plot_recovered_vs_simulated_activation(Time,TotalX,TotalU):
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize=[10,7])
    plt.suptitle("Simulated (gray) vs. Recovered Activations")
    for i in range(np.shape(TotalX)[0]):
        Activation1_Recovered,Activation2_Recovered = \
            return_muscle_activation_from_tension_and_muscle_length(
                Time,TotalX[i]
        )

        ax1.plot(Time,TotalU[i,0,:],'0.70',lw=3)
        ax1.plot(Time,Activation1_Recovered)
        ax2.plot(Time,TotalU[i,1,:],'0.70',lw=3)
        ax2.plot(Time,Activation2_Recovered)

    ax1.set_title("Muscle 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Activation")

    ax2.set_title("Muscle 2")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Activation")

    return(fig1)
