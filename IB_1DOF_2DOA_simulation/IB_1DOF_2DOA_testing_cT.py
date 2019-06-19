from pendulum_eqns.sim_eqns_ActIB_sinusoidal_activations import *
from recovered_tension import *
from recovered_muscle_length import *
from recovered_activation import *
from danpy.useful_functions import *
from danpy.sb import dsb,get_terminal_width
import pickle

X_o = np.array([r(0),dr(0)])
InitialTension = return_initial_tension(
    X_o,
    ReturnMultipleInitialTensions=False,
    Seed=2,
    Bounds = [[0,0.4*BIC.F_MAX],[0,0.4*TRI.F_MAX]],
    InitialAngularAcceleration=d2r(0),
    Return_k = False
)
InitialTensionVelocity = return_initial_tension_velocity(
    InitialTension,
    X_o,
    InitialAngularAcceleration=d2r(0),
    InitialAngularJerk=d2r(0),
    ParticularSolution=np.array([0,0])
)
InitialTensionAcceleration = return_initial_tension_acceleration(
    InitialTension,
    InitialTensionVelocity,
    X_o,
    InitialAngularAcceleration=d2r(0),
    InitialAngularJerk = d3r(0),
    InitialAngularSnap = d4r(0),
    ParticularSolution=np.array([0,0])
)

TerminalWidth = get_terminal_width()
####################################################
############  Running with Default cT  #############
####################################################

try:
    TensionTrialTitle = (
        "          Default cT          \n")
    print(
        " "*int(TerminalWidth/2 - len(TensionTrialTitle)/2)
        + colored(TensionTrialTitle,'red',attrs=["underline","bold"])
        )

    X,U = run_sim_IB_sinus_act(
        FixedInitialTension=InitialTension,
        Amp="Scaled",
        Freq=1,
        InitialAngularAcceleration=d2r(0),
        InitialAngularJerk=d3r(0),
        InitialAngularSnap=d4r(0),
        InitialTensionVelocity=InitialTensionVelocity,
        InititailTensionAcceleration=InitialTensionAcceleration,
        ICs = None
    )
    Period = 2*np.pi/Freq
    Modulo_IC_Indices = list(np.where(Time%Period==0)[0])[1:]

    if (X==np.zeros((8,len(Time)))).all():
        raise ValueError("Trial Failed... Try Again!")
except ValueError:
    print("Dont know what happened... Try Again!")
    raise

DefaultMuscleLength1 = X[4,:]
DefaultMuscleLength2 = X[5,:]

####################################################
###############  Sweeping Across cT  ###############
####################################################

count = 0
InitialTensionsFromSuccessfulTrials = []
dl_MTU1 = np.cumsum([BIC.v_MTU(X[:,j]) for j in range(len(Time))])*dt
dl_MTU2 = np.cumsum([TRI.v_MTU(X[:,j]) for j in range(len(Time))])*dt

cT_array = np.linspace(37.37-2*14.67,37.37+2*14.67,101)

MuscleLength1 = np.zeros((len(cT_array),len(Time)))
MuscleLength2 = np.zeros((len(cT_array),len(Time)))

statusbar = dsb(0,len(cT_array),title="Sweeping cT")
for i in range(len(cT_array)):
    MuscleLength1[i,:] = (
        (1/np.cos(BIC.pa))
        * dl_MTU1
        - (BIC.lTo*BIC.kT/np.cos(BIC.pa))*np.log(
            (np.exp(X[2,:]/(BIC.F_MAX*cT_array[i]*BIC.kT))-1)
            / (np.exp(X[2,0]/(BIC.F_MAX*cT_array[i]*BIC.kT))-1)
        )
        + X[4,0]
    )
    MuscleLength2[i,:] = (
        (1/np.cos(TRI.pa))
        * dl_MTU2
        - (TRI.lTo*TRI.kT/np.cos(TRI.pa))*np.log(
            (np.exp(X[3,:]/(TRI.F_MAX*cT_array[i]*TRI.kT))-1)
            / (np.exp(X[3,0]/(TRI.F_MAX*cT_array[i]*TRI.kT))-1)
        )
        + X[5,0]
    )
    statusbar.update(i)

standard_figs = plot_N_sim_IB_sinus_act(
    Time,X[np.newaxis,:,:],U[np.newaxis,:,:],
    Return=True,
    Normalized=True
)

cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(cT_array), vmax=max(cT_array))
colors = [cmap(normalize(value)) for value in cT_array]

fig1,ax1 = plt.subplots(1,figsize=(10,8))
fig2,ax2 = plt.subplots(1,figsize=(10,8))
fig3,ax3 = plt.subplots(1,figsize=(10,8))
fig4,ax4 = plt.subplots(1,figsize=(10,8))

for i in range(np.shape(MuscleLength1)[0]):
    ax1.plot(Time,(MuscleLength1[i,:]-DefaultMuscleLength1)/BIC.lo,c=colors[i])
    ax2.plot(Time,(MuscleLength2[i,:]-DefaultMuscleLength2)/TRI.lo,c=colors[i])
    ax3.plot(Time,MuscleLength1[i,:]/BIC.lo,c=colors[i])
    ax4.plot(Time,MuscleLength2[i,:]/TRI.lo,c=colors[i])

ax1.set_xlabel("Time (s)",fontsize=14)
ax1.set_ylabel("Norm. Error",fontsize=14)
ax1.set_title("Norm. Error vs. Time\n When Misjudging " + r"$c^T$" + "\n Muscle 1")
cax1, _ = matplotlib.colorbar.make_axes(ax1)
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=cmap, norm=normalize)
cax1.set_ylabel(r"$c^T$",fontsize=14)

ax2.set_xlabel("Time (s)",fontsize=14)
ax2.set_ylabel("Norm. Error",fontsize=14)
ax2.set_title("Norm. Error vs. Time\n When Misjudging " + r"$c^T$" + "\n Muscle 2")
cax2, _ = matplotlib.colorbar.make_axes(ax2)
cbar2 = matplotlib.colorbar.ColorbarBase(cax2, cmap=cmap, norm=normalize)
cax2.set_ylabel(r"$c^T$",fontsize=14)

ax3.set_xlabel("Time (s)",fontsize=14)
ax3.set_ylabel(r"$\hat{l}_m$",fontsize=14)
ax3.set_title("Norm. Muscle Length vs. Time\n When Sweeping " + r"$c^T$" + "\n Muscle 1")
cax3, _ = matplotlib.colorbar.make_axes(ax3)
cbar3 = matplotlib.colorbar.ColorbarBase(cax3, cmap=cmap, norm=normalize)
cax3.set_ylabel(r"$c^T$",fontsize=14)

ax4.set_xlabel("Time (s)",fontsize=14)
ax4.set_ylabel(r"$\hat{l}_m$",fontsize=14)
ax4.set_title("Norm. Muscle Length vs. Time\n When Sweeping " + r"$c^T$" + "\n Muscle 2")
cax4, _ = matplotlib.colorbar.make_axes(ax4)
cbar4 = matplotlib.colorbar.ColorbarBase(cax4, cmap=cmap, norm=normalize)
cax4.set_ylabel(r"$c^T$",fontsize=14)

# VelocityError1 = np.array(
#     list(
#         map(
#             lambda T,dT: (
#                 (BIC.lTo/BIC.lo)/(BIC.F_MAX*BIC.cT)
#                 * dT
#                 / (
#                     1
#                     - np.exp(-T/(BIC.F_MAX*BIC.cT*BIC.kT))
#                 )
#             ),
#             X[2,1:],
#             np.gradient(X[2,:],Time)
#         )
#     )
# )
# fig,[ax5,ax6] = plt.subplots(2,figsize=[12,7])
#
# ax5.scatter(X[6,5000:10001]/BIC.lo,VelocityError1[5000:10001],c=Time[5000:10001],cmap=cmap,norm=plt.Normalize(Time[5000],Time[10001]))
#
# ax6.scatter(Time[5000:10001],X[6,5000:10001]/BIC.lo,c=Time[5000:10001],cmap=cmap,norm=plt.Normalize(Time[5000],Time[10001]))
#
# ax5.set_xlabel(r"Norm. Muscle Velocity $\hat{l}_o/s$")
# ax5.set_ylabel(r"Norm. Velocity Error $\hat{l}_o/s$")
# ax6.set_xticks([Time[5001],Time[6251],Time[7501],Time[8751],Time[10001]])
# ax6.set_xticklabels([r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$",r"$\pi$"])
# ax6.set_xlabel("Phase")
# ax6.set_ylabel(r"Norm. Muscle Velocity $\hat{l}_o/s$")
# plt.show()

params["Muscle 1"] = BIC.__dict__
params["Muscle 1"]["Settings"] = BIC_Settings
params["Muscle 2"] = TRI.__dict__
params["Muscle 2"]["Settings"] = TRI_Settings

FilePath = save_figures(
    "output_figures/integrator_backstepping_testing_cT/",
    "v1.0",
    params,
    SaveAsPDF=True,
    ReturnPath=True
)

plt.close('all')
FormatedSaveData = {
    "Default X" : X,
    "Default U" : U,
    "Muscle 1 Lengths" : MuscleLength1,
    "Muscle 2 Lengths" : MuscleLength2,
    "Initial Tension" : InitialTension
}
pickle.dump(
    FormatedSaveData,
    open(
        FilePath+"/output.pkl",
        "wb"
        )
)
