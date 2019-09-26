import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from termcolor import cprint,colored
from danpy.sb import dsb,get_terminal_width
from pendulum_eqns.init_IB_sinusoid_model import *
### ONLY WORKS FOR REFERENCE TRAJECTORY 1

N_seconds = 4
N = N_seconds*5000 + 1
Time = np.linspace(0,N_seconds,N)
dt = Time[1]-Time[0]

def run_sim_IB_sinus_act(**kwargs):
    """
    Runs one simulation for INTEGRATOR BACKSTEPPING SINUSOIDAL INPUT control.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    1) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Tension_Bounds.

    2) InitialAngularAcceleration - must be a float or an int. Default is 0 (starting from rest).

    3) thresh - must be an int. Default is 25.

    4) FixedInitialTension - will be passed to find_viable_initial_values and will fix the value of initial tension. Must be a (2,) numpy.ndarray. Run find_initial_tension outside of the loop for a given seed and then feed it through the pipeline.

    5) Amps - list of length 2 that has the amplitudes of sinusoidal activation trajectories.

    6) Freq - scalar value given in Hz.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs (Passed to initialize_tendon_tension())
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    7) ReturnMultipleInitialTensions - must be either True or False. Default is False.

    8) Seed - must be a float or an int. Default is None (seeded by current time).

    9) Return - must be either true or false (default). When false, function will assign the initial tension value to self.

    10) TensionBounds - must be a list of size (2,2) or None (in which case the self.TensionBounds will be used)

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs (Passed to find_viable_initial_values())
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    11) FixedInitialMuscleLengths - must be a list of length 2 or None (Default). If is None, then find_viable_initial_values() will assign this value randomly. Used for trials where we wish to hold muscle length constant for different tension levels.

    12) Seed - must be a float or an int. Default is None (seeded by current time).

    """

    thresh = kwargs.get("thresh",25)
    assert type(thresh)==int, "thresh should be an int as it is the number of attempts the program should run before stopping."

    Bounds = kwargs.get("Bounds",Activation_Bounds)
    assert type(Bounds)==list and np.shape(Bounds)==(2,2), "Bounds should be a list of shape (2,2)."

    Amp = kwargs.get("Amp",1)
    if Amp is not "Scaled":
        assert type(Amp) in [int,float], "Amp should be an int or a float."

    Freq = kwargs.get("Freq",1)
    assert type(Freq) in [int,float], "Freq should be an int or a float."

    ICs = kwargs.get("ICs",None)

    TensionBounds = kwargs.get("TensionBounds",None)
    Seed = kwargs.get("Seed",None)

    AnotherIteration = True
    AttemptNumber = 1

    while AnotherIteration == True:
        Plant = Pendulum_1DOF_2DOA(BIC,TRI,Time)
        # should combine to one function now.
        Plant.initialize_tendon_tension(
            Seed=Seed,
            Bounds=TensionBounds
        )
        if ICs is None:
            Plant.find_viable_initial_values(**kwargs)
            Plant.set_X_o()
            Plant.set_U_o()
        else:
            assert ((len(ICs)==2)
                    and (np.shape(ICs[0])==(8,))
                    and (np.shape(ICs[1])==(2,))), \
                "ICs must be a list that contains X_o (of shape (8,)) and U_o (of shape (2,))."
            Plant.set_X_o(ICs[0])
            Plant.set_U_o(ICs[1])

        U = np.zeros((2,N))
        if Amp is "Scaled":
            Amp = 0.75*Plant.U_o[0]
            # Amp1 = 0.1
            # Amp2 = 0.50*InitialActivations[0]
            # assert InitialActivations[0]<0.15, "InitialActivations too high."
            # Amp2 = 0.6
        # assert (Amp1>=0) and (Amp2>=0), "Amp became negative. Run Again."
        assert Amp>0, "Amp became negative. Run Again."

        # phase1 = np.ones(np.shape(Time))*(np.array([np.sin(2*np.pi*Time)])>0)
        # phase2 = np.ones(np.shape(Time))*(np.array([np.sin(2*np.pi*Time)])<=0)
        # U[0,:] = (
        #     InitialActivations[0]
        #     + (Amp1*(1-np.cos(4*np.pi*Time))/2)*phase1
        #     + (Amp2*(1-np.cos(4*np.pi*Time))/2)*phase2
        # )
        # U[0,:] = U[0,:]*(U[0,:]>=0)
        # import ipdb; ipdb.set_trace()
        shift = 3*np.pi/16
        U[0,:] = (
            Plant.U_o[0]
            - Amp*np.sin(2*np.pi*Freq*Time+shift)
        )
        # U[0,:] = InitialActivations[0] + Amp*(1-np.cos(4*np.pi*Freq*Time))/2
        U[1,0] = Plant.U_o[1]
        U_o = U[:,0]
        #########
        # Plant = Pendulum_1DOF_2DOA(BIC,TRI,X_o,U_o,Time)
        Plant.U[0,:] = U[0,:]
        try:
            cprint("Attempt #" + str(int(AttemptNumber)) + ":\n", 'green')
            statusbar = dsb(0,N-1,title=run_sim_IB_sinus_act.__name__)
            for i in range(N-1):
                Plant.update_pendulum_variables(i)
                Plant.forward_integrate(i)
                statusbar.update(i)

            AnotherIteration = False
            return(Plant.X,Plant.U)
        except:
            print('\n')
            print(" "*(get_terminal_width()\
            			- len("...Attempt #" + str(int(AttemptNumber)) + " Failed. "))\
            			+ colored("...Attempt #" + str(int(AttemptNumber)) + " Failed. \n",'red'))
            AttemptNumber += 1
            if AttemptNumber > thresh:
                AnotherIteration=False
                return(np.zeros((8,N)),np.zeros((2,N)))

def run_N_sim_IB_sinus_act(**kwargs):
    """
    Runs one simulation for sinusoidal u1 control.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs (most are passed to run_sim_IB_sinus_act())
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    1) Bounds - must be a (2,2) list with each row in ascending order. Default is given by Tension_Bounds.

    2) InitialAngularAcceleration - must be a float or an int. Default is 0 (starting from rest).

    3) thresh - must be an int. Default is 25.

    4) FixedInitialTension - will be passed to find_viable_initial_values and will fix the value of initial tension. Must be a (2,) numpy.ndarray. Run find_initial_tension outside of the loop for a given seed and then feed it through the pipeline.

    5) Amps - list of length 2 that has the amplitudes of sinusoidal activation trajectories.

    6) Freq - scalar value given in Hz.

    7) PhaseOffset - scalar value in [0,360).

    8) InitialTensionAcceleration - will be passed to find_viable_initial_values(**kwargs). Must be a numpy array of shape (2,)

    9) NumberOfTrials - should be an int.

    10) FixedInitialMuscleLengths - must be a list of length 2 or None (Default). If is None, then program will assign this value randomly. Used for trials where we wish to hold muscle length constant for different tension levels. Will be passed to run_sim_IB_sinus_act(...)
    """
    NumberOfTrials = kwargs.get("NumberOfTrials",10)
    assert type(NumberOfTrials)==int and NumberOfTrials>0,"NumberOfTrials must be an positive int."

    TotalX = np.zeros((NumberOfTrials,8,N))
    TotalU = np.zeros((NumberOfTrials,2,N))
    TerminalWidth = get_terminal_width()

    print("\n")
    for j in range(NumberOfTrials):
        TrialTitle = (
            "          Trial "
            + str(j+1)
            + "/" +str(NumberOfTrials)
            + "          \n")
        print(
            " "*int(TerminalWidth/2 - len(TrialTitle)/2)
            + colored(TrialTitle,'white',attrs=["underline","bold"])
            )
        TotalX[j],TotalU[j] = run_sim_IB_sinus_act(**kwargs)
    i=0
    NumberOfSuccessfulTrials = NumberOfTrials
    while i < NumberOfSuccessfulTrials:
        if (TotalX[i]==np.zeros((8,np.shape(TotalX)[2]))).all():
            TotalX = np.delete(TotalX,i,0)
            TotalU = np.delete(TotalU,i,0)
            NumberOfSuccessfulTrials-=1
            if NumberOfSuccessfulTrials==0: raise ValueError("No Successful Trials!")
        else:
            i+=1

    print(
        "Number of Desired Runs: "
        + str(NumberOfTrials)
        + "\n"
        + "Number of Successful Runs: "
        + str(NumberOfSuccessfulTrials)
        + "\n"
    )
    return(TotalX,TotalU)

def plot_N_sim_IB_sinus_act(t,TotalX,TotalU,**kwargs):
    Return = kwargs.get("Return",False)
    assert type(Return) == bool, "Return should either be True or False"

    ReturnError = kwargs.get("ReturnError",False)
    assert type(ReturnError)==bool, "ReturnError should be either True or False."

    Normalized = kwargs.get("Normalized",False)
    assert type(Normalized)==bool, "Normalized should be either True or False."

    ######################################
    ###### Actual/Expected Endpoint ######
    ######################################
    fig1 = plt.figure(figsize = (9,7))
    fig1_title = "Underdetermined Forced-Pendulum Example"
    plt.title(fig1_title,fontsize=16,color='gray')
    statusbar = dsb(0,np.shape(TotalX)[0],title=(plot_N_sim_IB_sinus_act.__name__ + " (" + fig1_title +")"))
    for j in range(np.shape(TotalX)[0]):
        plt.plot(t,(TotalX[j,0,:])*180/np.pi,'0.70',lw=2)
        statusbar.update(j)
    plt.plot(np.linspace(0,t[-1],1001),\
        	(r(np.linspace(0,t[-1],1001)))*180/np.pi,\
               'r')
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Measure (Deg)")

    ######################################
    ########### Endpoint Error ###########
    ######################################
    fig2 = plt.figure(figsize = (9,7))
    fig2_title = "Error vs. Time"
    plt.title(fig2_title)
    statusbar.reset(title=(plot_N_sim_IB_sinus_act.__name__ + " (" + fig2_title +")"))
    for j in range(np.shape(TotalX)[0]):
        plt.plot(t, (r(t)-TotalX[j,0,:])*180/np.pi,color='0.70')
        statusbar.update(j)
    plt.xlabel("Time (s)")
    plt.ylabel("Error (Deg)")

    ######################################
    ############# plot_states ############
    ############# plot_inputs ############
    ######### plot_l_m_comparison ########
    ######################################
    statusbar.reset(
        title=(
        	plot_N_sim_IB_sinus_act.__name__
        	+ " (Plotting States, Inputs, and Muscle Length Comparisons)"
        	)
        )
    for j in range(np.shape(TotalX)[0]):
        if j == 0:
            fig3 = plot_states(
                t,TotalX[j],
                Return=True,
                InputString="Muscle Activations",
                Normalized=Normalized
            )
            fig4 = plot_inputs(
                t,TotalU[j],
                Return=True,
                InputString="Muscle Activations"
            )
            fig5,Error = plot_l_m_comparison(
                t,TotalX[j],
                MuscleLengths=TotalX[j,4:6,:],
                Return=True,
                InputString="Muscle Activation",
                Normalized=Normalized,
                ReturnError=True
            )
            Error1 = Error[0][np.newaxis,:]
            Error2 = Error[1][np.newaxis,:]
        else:
            fig3 = plot_states(
                t,TotalX[j],
                Return=True,
                InputString="Muscle Activations",
                Normalized=Normalized,
                Figure=fig3
            )
            fig4 = plot_inputs(
                t,TotalU[j],
                Return=True,
                InputString="Muscle Activations",
                Figure=fig4
            )
            fig5,Error = plot_l_m_comparison(
                t,TotalX[j],
                MuscleLengths=TotalX[j,4:6,:],
                Return=True,
                InputString="Muscle Activation",
                Normalized=Normalized,
                ReturnError=True,
                Figure=fig5
            )
            Error1 = np.concatenate([Error1,Error[0][np.newaxis,:]],axis=0)
            Error2 = np.concatenate([Error2,Error[1][np.newaxis,:]],axis=0)
        statusbar.update(j)

    if Return == True:
        if ReturnError == True:
            return([fig1,fig2,fig3,fig4,fig5],[-Error1,-Error2])
        else:
            return([fig1,fig2,fig3,fig4,fig5])
    else:
        if ReturnError == True:
            plt.show()
            return([-Error1,-Error2])
        else:
            plt.show()

def plot_l_m_approximation_error_vs_tendon_tension(t,TotalX,Error,**kwargs):

    Return = kwargs.get("Return",False)
    assert type(Return) == bool, "Return should either be True or False"

    InitialTensions = kwargs.get("InitialTensions",[TotalX[0,2:4,0]])
    assert type(InitialTensions)==list,"InitialTensions must be a list or arrays"
    assert all(np.array([str(type(el))=="<class 'numpy.ndarray'>" for el in InitialTensions])), "All elements of InitialTensions must be a numpy.ndarray."

    NumberOfTensionTrials = len(InitialTensions)
    TendonTension1 = np.linspace(0.01*F_MAX1,0.9*F_MAX1,1001)
    TendonTension2 = np.linspace(0.01*F_MAX2,0.9*F_MAX2,1001)

    fig1,axes1 = plt.subplots(2,2,figsize=(10,8))
    plt.suptitle("Error from MTU Approx vs. Tendon Tension\nMuscle 1",fontsize=16)
    axes1[0][0].set_xlabel("Tendon Tension (N)")
    axes1[0][0].set_ylabel("Error (m)")
    axes1[0][0].set_xlim(
            TotalX[:,2,:].min()-0.1*(TotalX[:,2,:].max()-TotalX[:,2,:].min()),
            TotalX[:,2,:].max()+0.1*(TotalX[:,2,:].max()-TotalX[:,2,:].min()))
    axes1[0][0].set_ylim(
            Error[0].min()-0.1*(Error[0].max()-Error[0].min()),
            Error[0].max()+0.1*(Error[0].max()-Error[0].min()))
    # axes1[0][0].plot(TendonTension1,Error1,'0.70',lw=2)
    axes1[0][1].set_xlabel(r"$\longrightarrow$ Time (s) $\longrightarrow$")
    axes1[0][1].set_ylim(axes1[0][0].get_ylim())
    axes1[0][1].set_yticklabels(["" for el in axes1[0][1].get_yticks()])
    axes1[1][0].set_ylabel(r"$\longleftarrow$ Time (s) $\longleftarrow$")
    axes1[1][0].set_xlim(axes1[0][0].get_xlim())
    axes1[1][0].set_xticklabels(["" for el in axes1[0][0].get_xticks()])
    axes1[1][0].yaxis.tick_right()
    axes1[1][0].yaxis.set_label_position("right")
    axes1[1][0].set_yticks(-np.array(list(range(N_seconds+1))))
    axes1[1][0].set_yticklabels([str(-el) for el in axes1[1][0].get_yticks()])
    axes1[1][1].text(0.00,0.65,
        (r'error $= \frac{\tau}{k}\cdot\ln\left(\frac{e^{T_{1}(t)/\tau} - 1}{e^{T_{1}(0)/\tau} - 1} \right )$'),fontsize=20)
    axes1[1][1].text(0.075,0.475,
        (r'          - $(1 - \cos(\alpha_{1}))\left[l_{m,1}(t) - l_{m,1}(0) \right]$'), fontsize=16)
    axes1[1][1].text(0.15,0.325,
        (r'where,    $\tau = F_{MAX,1}\cdot c^T \cdot k^T$'),fontsize=14)
    axes1[1][1].text(0.15,0.15,
        (r'and    $k = \frac{F_{MAX,1}\cdot c^T}{l_{T_{o,1}}}$'),fontsize=14)
    axes1[1][1].axis('off')

    fig2,axes2 = plt.subplots(2,2,figsize=(10,8))
    plt.suptitle("Error from MTU Approx vs. Tendon Tension\nMuscle 2",fontsize=16)
    axes2[0][0].set_ylabel("Error (m)")
    axes2[0][0].set_xlabel("Tendon Tension (N)")
    axes2[0][0].set_xlim(
            TotalX[:,3,:].min()-0.1*(TotalX[:,3,:].max()-TotalX[:,3,:].min()),
            TotalX[:,3,:].max()+0.1*(TotalX[:,3,:].max()-TotalX[:,3,:].min()))
    axes2[0][0].set_ylim(
            Error[1].min()-0.1*(Error[1].max()-Error[1].min()),
            Error[1].max()+0.1*(Error[1].max()-Error[1].min()))
    # axes2[0][0].plot(TendonTension2,Error2,'0.70',lw=2)
    axes2[0][1].set_xlabel(r"$\longrightarrow$ Time (s) $\longrightarrow$")
    axes2[0][1].set_ylim(axes2[0][0].get_ylim())
    axes2[0][1].set_yticklabels(["" for el in axes2[0][1].get_yticks()])
    axes2[1][0].set_ylabel(r"$\longleftarrow$ Time (s) $\longleftarrow$")
    axes2[1][0].set_xlim(axes2[0][0].get_xlim())
    axes2[1][0].set_xticklabels(["" for el in axes2[0][0].get_xticks()])
    axes2[1][0].yaxis.tick_right()
    axes2[1][0].yaxis.set_label_position("right")
    axes2[1][0].set_yticks(-np.array(list(range(N_seconds+1))))
    axes2[1][0].set_yticklabels([str(-el) for el in axes1[1][0].get_yticks()])
    axes2[1][1].text(0.00,0.65,
        (r'error $= \frac{\tau}{k}\cdot\ln\left(\frac{e^{T_{2}(t)/\tau} - 1}{e^{T_{2}(0)/\tau} - 1} \right )$'),fontsize=20)
    axes2[1][1].text(0.075,0.475,
        (r'          - $(1 - \cos(\alpha_{2}))\left[l_{m,2}(t) - l_{m,2}(0) \right]$'), fontsize=16)
    axes2[1][1].text(0.15,0.325,
        (r'where,    $\tau = F_{MAX,2}\cdot c^T \cdot k^T$'),fontsize=14)
    axes2[1][1].text(0.15,0.15,
        (r'and    $k = \frac{F_{MAX,2}\cdot c^T}{l_{T_{o,2}}}$'),fontsize=14)
    axes2[1][1].axis('off')

    for i in range(NumberOfTensionTrials):
        error_function_1 = return_error_func_no_pennation(InitialTensions[i][0],F_MAX1,lTo1)
        error_function_2 = return_error_func_no_pennation(InitialTensions[i][1],F_MAX2,lTo2)
        Error1 = error_function_1(TendonTension1)
        Error2 = error_function_2(TendonTension2)
        axes1[0][0].plot(TendonTension1,Error1,str(1-InitialTensions[i][0]/F_MAX1),lw=2)
        axes2[0][0].plot(TendonTension2,Error2,str(1-InitialTensions[i][1]/F_MAX2),lw=2)

    statusbar = dsb(0,np.shape(TotalX)[0],
        title=plot_l_m_approximation_error_vs_tendon_tension.__name__)
    for i in range(np.shape(TotalX)[0]):
        axes1[0][0].plot(TotalX[i,2,:],Error[0][i])
        axes1[0][1].plot(Time,Error[0][i])
        axes1[1][0].plot(TotalX[i,2,:],-Time)

        axes2[0][0].plot(TotalX[i,3,:],Error[1][i])
        axes2[0][1].plot(Time,Error[1][i])
        axes2[1][0].plot(TotalX[i,3,:],-Time)
        statusbar.update(i)

    if Return == True:
        return([fig1,fig2])
    else:
        plt.show()

def plot_l_m_error_manifold(t,TotalX,Error,**kwargs):

    Return = kwargs.get("Return",False)
    assert type(Return) == bool, "Return should either be True or False"

    InitialTensions = kwargs.get("InitialTensions",[TotalX[0,2:4,0]])
    assert type(InitialTensions)==list,"InitialTensions must be a list or arrays"
    assert all(np.array([str(type(el))=="<class 'numpy.ndarray'>" for el in InitialTensions])), "All elements of InitialTensions must be a numpy.ndarray."

    NumberOfTensionTrials = len(InitialTensions)

    fig1 = plt.figure(figsize=(10,8))
    axes1_1 = fig1.add_subplot(221, projection='3d')
    axes1_2 = fig1.add_subplot(222)
    axes1_3 = fig1.add_subplot(223)
    axes1_4 = fig1.add_subplot(224)

    plt.suptitle("Error from MTU Approx vs. Tendon Tension\nMuscle 1",fontsize=16)

    fig2 = plt.figure(figsize=(10,8))
    axes2_1 = fig2.add_subplot(221, projection='3d')
    axes2_2 = fig2.add_subplot(222)
    axes2_3 = fig2.add_subplot(223)
    axes2_4 = fig2.add_subplot(224)

    plt.suptitle("Error from MTU Approx vs. Tendon Tension\nMuscle 2",fontsize=16)

    statusbar = dsb(0,np.shape(TotalX)[0],
        title=plot_l_m_approximation_error_vs_tendon_tension.__name__)
    for i in range(np.shape(TotalX)[0]):
        axes1_1.plot(TotalX[i,4,:],TotalX[i,2,:],Error[0][i])
        axes1_2.plot(Time,Error[0][i])
        axes1_3.plot(TotalX[i,2,:],-Time)

        axes2_1.plot(TotalX[i,5,:],TotalX[i,3,:],Error[1][i])
        axes2_2.plot(Time,Error[1][i])
        axes2_3.plot(TotalX[i,3,:],-Time)
        statusbar.update(i)

    for i in range(TotalX.shape[0]):
        error_function_1 = \
                return_error_func(TotalX[i,2,0],TotalX[i,4,0],F_MAX1,lTo1,α1)
        error_function_2 = \
                return_error_func(TotalX[i,3,0],TotalX[i,5,0],F_MAX2,lTo2,α2)

        MinimumTension1 = TotalX[:,2,:].min()
        MaximumTension1 = TotalX[:,2,:].max()
        Tension1Range = TotalX[:,2,:].max() - TotalX[:,2,:].min()
        TendonTension1 = np.linspace(
                    MinimumTension1 - 0.05*Tension1Range,
                    MaximumTension1 + 0.05*Tension1Range,
                    1001
                    )

        MinimumMuscleLength1 = TotalX[:,4,:].min()
        MaximumMuscleLength1 = TotalX[:,4,:].max()
        MuscleLength1Range = TotalX[:,4,:].max() - TotalX[:,4,:].min()
        MuscleLength1 = np.linspace(
                    MinimumMuscleLength1 - 0.05*MuscleLength1Range,
                    MaximumMuscleLength1 + 0.05*MuscleLength1Range,
                    1001
                    )

        MuscleLength1Mesh, TendonTension1Mesh = \
                np.meshgrid(MuscleLength1,TendonTension1)
        Error1 = \
                error_function_1(TendonTension1Mesh,MuscleLength1Mesh)

        MinimumTension2 = TotalX[:,3,:].min()
        MaximumTension2 = TotalX[:,3,:].max()
        Tension2Range = TotalX[:,3,:].max() - TotalX[:,3,:].min()
        TendonTension2 = np.linspace(
                    MinimumTension2 - 0.05*Tension2Range,
                    MaximumTension2 + 0.05*Tension2Range,
                    1001
                    )

        MinimumMuscleLength2 = TotalX[:,5,:].min()
        MaximumMuscleLength2 = TotalX[:,5,:].max()
        MuscleLength2Range = TotalX[:,5,:].max() - TotalX[:,5,:].min()
        MuscleLength2 = np.linspace(
                    MinimumMuscleLength2 - 0.05*MuscleLength2Range,
                    MaximumMuscleLength2 + 0.05*MuscleLength2Range,
                    1001
                    )

        MuscleLength2Mesh, TendonTension2Mesh = \
                np.meshgrid(MuscleLength2,TendonTension2)
        Error2 = \
                error_function_2(TendonTension2Mesh,MuscleLength2Mesh)

        axes1_1.plot_surface(MuscleLength1Mesh,
                            TendonTension1Mesh,
                            Error1,
                            color=str(np.linspace(0.25,0.75,TotalX.shape[0])[i]))
        axes2_1.plot_surface(MuscleLength2Mesh,
                            TendonTension2Mesh,
                            Error2,
                            color=str(np.linspace(0.25,0.75,TotalX.shape[0])[i]))

    axes1_1.set_xlabel("Muscle Length (m)")
    axes1_1.set_ylabel("Tendon Tension (N)")
    axes1_1.set_zlabel("Error (m)")
    axes1_2.set_xlabel(r"$\longrightarrow$ Time (s) $\longrightarrow$")
    axes1_2.set_ylim(axes1_1.get_zlim())
    # axes1_2.set_yticklabels(["" for el in axes1_2.get_yticks()])
    axes1_3.set_ylabel(r"$\longleftarrow$ Time (s) $\longleftarrow$")
    axes1_3.set_xlim(axes1_1.get_ylim())
    # axes1_3.set_xticklabels(["" for el in axes1_1.get_xticks()])
    axes1_3.yaxis.tick_right()
    axes1_3.yaxis.set_label_position("right")
    axes1_3.set_yticks(-np.array(list(range(N_seconds+1))))
    axes1_3.set_yticklabels([str(-el) for el in axes1_3.get_yticks()])
    axes1_4.text(0.00,0.65,
        (r'error $= \frac{\tau}{k}\cdot\ln\left(\frac{e^{T_{1}(t)/\tau} - 1}{e^{T_{1}(0)/\tau} - 1} \right )$'),fontsize=20)
    axes1_4.text(0.075,0.475,
        (r'          - $(1 - \cos(\alpha_{1}))\left[l_{m,1}(t) - l_{m,1}(0) \right]$'), fontsize=16)
    axes1_4.text(0.15,0.325,
        (r'where,    $\tau = F_{MAX,1}\cdot c^T \cdot k^T$'),fontsize=14)
    axes1_4.text(0.15,0.15,
        (r'and    $k = \frac{F_{MAX,1}\cdot c^T}{l_{T_{o,1}}}$'),fontsize=14)
    axes1_4.axis('off')

    axes2_1.set_xlabel("Muscle Length (m)")
    axes2_1.set_ylabel("Tendon Tension (N)")
    axes2_1.set_zlabel("Error (m)")
    axes2_2.set_xlabel(r"$\longrightarrow$ Time (s) $\longrightarrow$")
    axes2_2.set_ylim(axes2_1.get_zlim())
    # axes2_2.set_yticklabels(["" for el in axes2_2.get_yticks()])
    axes2_3.set_ylabel(r"$\longleftarrow$ Time (s) $\longleftarrow$")
    axes2_3.set_xlim(axes2_1.get_ylim())
    # axes2_3.set_xticklabels(["" for el in axes2_1.get_xticks()])
    axes2_3.yaxis.tick_right()
    axes2_3.yaxis.set_label_position("right")
    axes2_3.set_yticks(-np.array(list(range(N_seconds+1))))
    axes2_3.set_yticklabels([str(-el) for el in axes1_3.get_yticks()])
    axes2_4.text(0.00,0.65,
        (r'error $= \frac{\tau}{k}\cdot\ln\left(\frac{e^{T_{2}(t)/\tau} - 1}{e^{T_{2}(0)/\tau} - 1} \right )$'),fontsize=20)
    axes2_4.text(0.075,0.475,
        (r'          - $(1 - \cos(\alpha_{2}))\left[l_{m,2}(t) - l_{m,2}(0) \right]$'), fontsize=16)
    axes2_4.text(0.15,0.325,
        (r'where,    $\tau = F_{MAX,2}\cdot c^T \cdot k^T$'),fontsize=14)
    axes2_4.text(0.15,0.15,
        (r'and    $k = \frac{F_{MAX,2}\cdot c^T}{l_{T_{o,2}}}$'),fontsize=14)
    axes2_4.axis('off')

    if Return == True:
        return([fig1,fig2])
    else:
        plt.show()

def return_error(T,l_m,F_MAX,lTo,α):
    tau = F_MAX*cT*kT
    alpha = F_MAX*cT/lTo
    error = (tau/alpha)*np.log((np.exp(T/tau) - 1)/(np.exp(T[0]/tau) - 1)) \
                + (np.cos(α) - 1)*(l_m - l_m[0])
    return(error)

def return_error_func(T_o,l_mo,F_MAX,lTo,α):
    tau = F_MAX*cT*kT
    alpha = F_MAX*cT/lTo
    def error_func(T,l_m):
        return((tau/alpha)*np.log((np.exp(T/tau) - 1)/(np.exp(T_o/tau) - 1))
                    + (np.cos(α) - 1)*(l_m - l_mo))
    return(error_func)

def return_error_no_pennation(T,F_MAX,lTo):
    tau = F_MAX*cT*kT
    alpha = F_MAX*cT/lTo
    error = (tau/alpha)*np.log((np.exp(T/tau) - 1)/(np.exp(T[0]/tau) - 1))
    return(error)

def return_error_func_no_pennation(T_o,F_MAX,lTo):
    tau = F_MAX*cT*kT
    alpha = F_MAX*cT/lTo
    def error_func(T):
        return((tau/alpha)*np.log((np.exp(T/tau) - 1)/(np.exp(T_o/tau) - 1)))
    return(error_func)
