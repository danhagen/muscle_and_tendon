import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from danpy.sb import dsb
from scipy.optimize import fsolve
import random


l_Ts_over_l_To = 0.95
initial_force = 0.01
kT_intersection = (
    (l_Ts_over_l_To - 1)
    / (
        np.log(
            np.exp(initial_force/0.20)
            - 1
        )
        - 1/0.20
    )
)
max_cT = 100

cT = 40
LrT = 1 - 1/cT
kT = np.linspace(0.0001,0.2/cT,1001)
lT = np.linspace(1,1.03,1001)
l_Ts_over_l_To = 0.95

cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(kT), vmax=max(kT))
colors = [cmap(normalize(value)) for value in kT]
max_curvature = np.zeros(np.shape(kT))
estimated_curvature = np.zeros(np.shape(kT))
fig1 = plt.figure()
ax1 = plt.gca()
fig2 = plt.figure()
ax2 = plt.gca()
fig3 = plt.figure()
ax3 = plt.gca()
statusbar = dsb(0,len(kT),title="Estimating Curvature")
for i in range(len(kT)):
    y = cT*kT[i]*np.log(np.exp((l_Ts_over_l_To*lT-LrT)/kT[i])+1)
    y_prime = (l_Ts_over_l_To)*cT/(1 + np.exp((LrT-l_Ts_over_l_To*lT)/kT[i]))
    y_dbl_prime = (cT/kT[i])*np.exp((LrT-l_Ts_over_l_To*lT)/kT[i])/((1+np.exp((LrT-l_Ts_over_l_To*lT)/kT[i]))**2)*l_Ts_over_l_To**2
    curvature = y_dbl_prime/((1+y_prime**2)**(3/2))
    max_curvature[i] = curvature[np.where(curvature == max(curvature))[0][0]]
    estimated_curvature[i] = l_Ts_over_l_To/kT[i]*2*np.sqrt(3)/9
    ax1.plot(lT,curvature,c=colors[i])
    ax2.plot(lT,y,c=colors[i])
    ax3.scatter([estimated_curvature[i]],[max_curvature[i]],c=[colors[i]])
    statusbar.update(i)
ax1.set_title("Sweeping " + r"$k^T$")
ax1.set_xlabel(r"$\hat{l}_T$")
ax1.set_ylabel(r"$\kappa$")
cax1, _ = matplotlib.colorbar.make_axes(ax1)
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=cmap)
cax1.set_ylabel(r"$k_T$",fontsize=14)
cbar1.set_ticks([0,0.25,0.5,0.75,1.0])
cbar1_labels = ["{:.5f}".format(el) for el in np.linspace(0,kT[-1],5)]
cax1.set_yticklabels(cbar1_labels)

ax2.set_title("Sweeping " + r"$k^T$")
ax2.set_xlabel(r"$\hat{l}_T$")
ax2.set_ylabel(r"$\hat{f}_T$")
cax2, _ = matplotlib.colorbar.make_axes(ax2)
cbar2 = matplotlib.colorbar.ColorbarBase(cax2, cmap=cmap)
cax2.set_ylabel(r"$k_T$",fontsize=14)
cbar2.set_ticks([0,0.25,0.5,0.75,1.0])
cbar2_labels = ["{:.5f}".format(el) for el in np.linspace(0,kT[-1],5)]
cax2.set_yticklabels(cbar2_labels)

ax3.set_title("Sweeping " + r"$k^T$")
ax3.set_xlabel(r"$\kappa_{approx}$")
ax3.set_ylabel(r"$\kappa_{actual}$")
cax3, _ = matplotlib.colorbar.make_axes(ax3)
cbar3 = matplotlib.colorbar.ColorbarBase(cax3, cmap=cmap)
cax3.set_ylabel(r"$k_T$",fontsize=14)
cbar3.set_ticks([0,0.25,0.5,0.75,1.0])
cbar3_labels = ["{:.5f}".format(el) for el in np.linspace(0,kT[-1],5)]
cax3.set_yticklabels(cbar3_labels)
ax3.plot([0,max(max_curvature)],[0,max(max_curvature)],'k--')

ratio = []
CT = []
KT = []
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
cT = np.linspace(10,100,101)

Seed = None
np.random.seed(Seed)
random.seed(Seed)
numTrials = 10000
count = 0
breakcount = 0
KT = []
CT = []
ratio = []
lT = np.linspace(1,1.03,1001)
while count<numTrials:
    try:
        kT_rand = random.uniform(0.0001,kT_intersection)
        if kT_rand <= 0.20/max_cT:
            ub_cT = max_cT
        else:
            ub_cT = 0.20/kT_rand
        if kT_rand==0:
            lb_cT = (1-initial_force)/(1-l_Ts_over_l_To)
        else:
            try:
                lb_cT = fsolve(
                    lambda cT: (
                        (1/l_Ts_over_l_To)*(
                            kT_rand*np.log(
                                np.exp(initial_force/(cT*kT_rand))
                                - 1
                            )
                            + (1-1/cT)
                        )
                        - 1
                    ),
                    20
                )[0]
            except:
                pass
        cT_rand = random.uniform(lb_cT,ub_cT)
        y_prime = (l_Ts_over_l_To)*cT_rand/(1 + np.exp((1-1/cT_rand-l_Ts_over_l_To*lT)/kT_rand))
        y_dbl_prime = (cT_rand/kT_rand)*np.exp((1-1/cT_rand-l_Ts_over_l_To*lT)/kT_rand)/((1+np.exp((1-1/cT_rand-l_Ts_over_l_To*lT)/kT_rand))**2)*l_Ts_over_l_To**2
        curvature = y_dbl_prime/((1+y_prime**2)**(3/2))
        max_curvature = curvature[np.where(curvature == max(curvature))[0][0]]
        estimated_curvature = l_Ts_over_l_To/kT_rand*2*np.sqrt(3)/9
        if estimated_curvature/max_curvature<1.05:
            ratio.append(estimated_curvature/max_curvature)
            KT.append(kT_rand)
            CT.append(cT_rand)
            count+=1
    except:
        breakcount+=1
    if breakcount>=1000:
        import ipdb; ipdb.set_trace()
# for j in range(len(cT)):
#     kT = np.linspace(0.0001,0.20/cT[j],101)
#     for i in range(len(kT)):
#         y_prime = (l_Ts_over_l_To)*cT[j]/(1 + np.exp((LrT-l_Ts_over_l_To*lT)/kT[i]))
#         y_dbl_prime = (cT[j]/kT[i])*np.exp((LrT-l_Ts_over_l_To*lT)/kT[i])/((1+np.exp((LrT-l_Ts_over_l_To*lT)/kT[i]))**2)*l_Ts_over_l_To**2
#         curvature = y_dbl_prime/((1+y_prime**2)**(3/2))
#         max_curvature = curvature[np.where(curvature == max(curvature))[0][0]]
#         estimated_curvature = l_Ts_over_l_To/kT[i]*2*np.sqrt(3)/9
#         ratio.append(estimated_curvature/max_curvature)
#         CT.append(cT[j])
#         KT.append(kT[i])
#     statusbar.update(j)

normalize = matplotlib.colors.Normalize(vmin=min(ratio), vmax=max(ratio))
colors = [cmap(normalize(value)) for value in ratio]
ax4.scatter(CT,KT,ratio,c=colors)
ax4.set_title("Sweeping " + r"$k^T$" + " and " + r"$c^T$")
ax4.set_zlabel(r"$\kappa_{approx}/\kappa_{actual}$")
ax4.set_ylabel(r"$k^T$")
ax4.set_xlabel(r"$c^T$")
ax4.set_xticks([10,20,30,40,50,60,70,80,90,100])
ax4.set_yticks(np.linspace(0,max(KT),5))

plt.show()
