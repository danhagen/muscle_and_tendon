from pendulum_eqns.sim_eqns_ActIB_sinusoidal_activations import *
from recovered_tension import *
from recovered_muscle_length import *
from recovered_activation import *
from useful_functions import *
import pickle

X_o = np.array([r(0),dr(0)])
InitialTensions = return_initial_tension(
                        X_o,
                        ReturnMultipleInitialTensions=True,
                        Bounds = [[0,0.4*F_MAX1],[0,0.4*F_MAX2]],
                        InitialAngularAcceleration=0
                        ) # list of len::8
InitialTensions = InitialTensions[3:6]
NumberOfTensionTrials = len(InitialTensions)
InitialTensionsFromSuccessfulTrials = []
TerminalWidth = get_terminal_width()
count = 0
for i in range(NumberOfTensionTrials):
    try:
        TensionTrialTitle = (
            "          Tension Setting "
            + str(i+1)
            + "/" +str(NumberOfTensionTrials)
            + "          \n")
        print(
        	" "*int(TerminalWidth/2 - len(TensionTrialTitle)/2)
        	+ colored(TensionTrialTitle,'blue',attrs=["underline","bold"])
        	)

        TotalX_temp,TotalU_temp = run_N_sim_IB_sinus_act(
                NumberOfTrials=1,
                FixedInitialTension=InitialTensions[i],
                Amp="Scaled",
                Freq=1,
                InitialAngularAcceleration=0,
                InitialAngularSnap=0
                )
        plt.close('all')
        count+=1

        if count == 1:
            TotalX = TotalX_temp
            TotalU = TotalU_temp
            InitialTensionsFromSuccessfulTrials.append(TotalX_temp[0,2:4,0])
        else:
            TotalX = np.concatenate([TotalX,TotalX_temp],axis=0)
            TotalU = np.concatenate([TotalU,TotalU_temp],axis=0)
            InitialTensionsFromSuccessfulTrials.append(TotalX_temp[0,2:4,0])
    except:
        print("Trial " + str(i+1) + " Failed...")

print("Number of Total Trials: " + str(NumberOfTensionTrials) + "\n")
print(
    "Number of Successful Trials: "
    + str(len(InitialTensionsFromSuccessfulTrials))
    )

if len(InitialTensions) != 0:
    figs = plot_N_sim_IB_sinus_act(
        Time,TotalX,TotalU,
        Return=True
    )
    recoverd_tension_figs = plot_recovered_vs_simulated_tension(
        Time,TotalX
    )
    recoverd_muscle_length_figs = \
        plot_recovered_vs_simulated_muscle_length(
            Time,TotalX
    )
    recoverd_muscle_activation_figs = \
        plot_recovered_vs_simulated_activation(
            Time,TotalX,TotalU
    )
    # plt.show()

    FilePath = save_figures(
        "output_figures/integrator_backstepping_sinusoidal_activations/",
        "1DOF_2DOA_v1.0",
        SaveAsPDF=True,
        ReturnPath=True
    )
    plt.close('all')
    FormatedSaveData = {
            "States" : TotalX,
            "Input" : TotalU,
            "Initial Tensions" : InitialTensionsFromSuccessfulTrials
            }
    pickle.dump(
        FormatedSaveData,
        open(
            FilePath+"/output.pkl",
            "wb"
            )
        )
else:
    print("All Trials Unsuccessful...")
