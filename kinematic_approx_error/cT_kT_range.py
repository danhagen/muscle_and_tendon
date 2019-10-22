import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from danpy.useful_functions import save_figures

#########################################
#### Finding the bounding conditions ####
#########################################
initial_force = 0.001

l_To_over_l_Ts_min = 1.03
l_To_over_l_Ts_max = 1.07
l_Ts_over_l_To_array = np.linspace(
    1/l_To_over_l_Ts_max,
    1/l_To_over_l_Ts_min,
    10
)

kT_min = 0
kT_max = 0
cT_min = 30
cT_max = 0

kT_array_lb = np.zeros(1001)
kT_array_ub = np.zeros(1001)
cT_array_lb = np.zeros(1001)
cT_array_ub = np.zeros(1001)

cT_intersection_min = 50
cT_intersection_max = 0
cT_baseline_min = 50
cT_baseline_max = 0

for i in range(len(l_Ts_over_l_To_array)):
    kT_intersection =(
        (l_Ts_over_l_To_array[i] - 1)
        / (
            np.log(
                np.exp(initial_force/0.20)
                - 1
            )
            - 1/0.20
        )
    )
    kT_max = max(
        kT_intersection,
        kT_max
    )
    kT_array = np.linspace(0,kT_intersection,1001)
    if i==0:
        kT_array_lb = kT_array
    elif i==(len(l_Ts_over_l_To_array)-1):
        kT_array_ub = kT_array
    for j in range(len(kT_array)):
        kT = kT_array[j]
        if kT==0:
            cT = (1-initial_force)/(1-l_Ts_over_l_To_array[i])
            cT_baseline_min = min(
                cT,
                cT_baseline_min
            )
            cT_baseline_max = max(
                cT,
                cT_baseline_max
            )
        else:
            cT = fsolve(
                lambda cT: (
                    (1/l_Ts_over_l_To_array[i])*(
                        kT*np.log(
                            np.exp(initial_force/(cT*kT))
                            - 1
                        )
                        + (1-1/cT)
                    )
                    - 1
                ),
                20
            )[0]
        if i==0:
            cT_array_lb[j] = cT
        elif i==(len(l_Ts_over_l_To_array)-1):
            cT_array_ub[j] = cT
        cT_min = min(
            cT,
            cT_min
        )
        cT_max = max(
            cT,
            cT_max
        )
        if j==len(kT_array)-1:
            cT_intersection_min = min(
                cT_intersection_min,
                cT
            )
            cT_intersection_max = max(
                cT_intersection_max,
                cT
            )

cT_range = cT_max-cT_min
kT_range = kT_max - kT_min
cT_array = np.linspace(
    cT_min - 0.05*cT_range,
    cT_max + 0.05*cT_range,
    101
)
kT_array = np.linspace(
    kT_min - 0.05*kT_range,
    kT_max + 0.05*kT_range,
    101
)

CT,KT = np.meshgrid(cT_array,kT_array)


cmap = matplotlib.cm.get_cmap('viridis')
normalize_to_tendon_opt_slack_ratio = matplotlib.colors.Normalize(
    vmin=min(1/l_Ts_over_l_To_array),
    vmax=max(1/l_Ts_over_l_To_array)
)


fig = plt.figure()
ax = plt.gca()
ax.set_xlim(cT_min,cT_max)
ax.set_xlabel(r"$c^T$",fontsize=14)
ax.set_ylim(kT_min,kT_max)
ax.set_ylabel(r"$k^T$",fontsize=14)
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap)
cax.set_ylabel(r"$\dfrac{l_{T,o}}{l_{T,s}}$",fontsize=14)
cbar.set_ticks(
    [
        (
            int(
                100*(el-np.floor(min(1/l_Ts_over_l_To_array)/0.01)*0.01)
                / (
                    np.ceil(max(1/l_Ts_over_l_To_array)/0.01)*0.01
                    - np.floor(min(1/l_Ts_over_l_To_array)/0.01)*0.01
                )
            )/100
        )
        for el in np.arange(
            np.floor(min(1/l_Ts_over_l_To_array)/0.01)*0.01,
            np.ceil(max(1/l_Ts_over_l_To_array)/0.01)*0.01+1e-4,
            0.01
        )
    ]
)
cax.set_yticklabels(
    [
        '{:0.2f}'.format(el)
        for el in np.arange(
            np.floor(min(1/l_Ts_over_l_To_array)/0.01)*0.01,
            np.ceil(max(1/l_Ts_over_l_To_array)/0.01)*0.01+1e-4,
            0.01
        )
    ]
)

for i in range(len(kT_array)-1):
    for j in range(len(cT_array)-1):
        kT = np.average([KT[i,j],KT[i+1,j]])
        cT = np.average([CT[i,j],CT[i,j+1]])
        l_Ts_over_l_To = (
            kT*np.log(
                np.exp(initial_force/(cT*kT))
                - 1
            )
            + (1 - 1/cT)
        )
        verts = np.array(
            [
                [CT[i,j],KT[i,j]],
                [CT[i,j+1],KT[i,j]],
                [CT[i,j+1],KT[i+1,j]],
                [CT[i,j],KT[i+1,j]]
            ]
        )
        ax.add_patch(
            matplotlib.patches.Polygon(
                verts,
                fc=cmap(normalize_to_tendon_opt_slack_ratio(1/l_Ts_over_l_To)),
                alpha=0.75
            )
        )


constraint_cT = np.concatenate(
    [
        cT_array_lb,
        np.linspace(
            cT_intersection_min,
            cT_intersection_max,
            1001
        ),
        np.array(list(reversed(cT_array_ub))),
        np.linspace(cT_baseline_max,cT_baseline_min,1001)
    ]
)

constraint_kT = np.concatenate(
    [
        kT_array_lb,
        0.2/np.linspace(
            cT_intersection_min,
            cT_intersection_max,
            1001
        ),
        np.array(list(reversed(kT_array_ub))),
        np.zeros(1001)
    ]
)

ax.plot(constraint_cT,constraint_kT,'k')

# save_figures(
#     "./Figures/feasible_cTkT/",
#     "cT_kT_range",
#     {},
#     figs=[fig],
#     SaveAsPDF=True
# )
plt.close(fig)

fig2 = plt.figure(figsize=(10,3))
stuff = pickle.load(open("Figures/Iso_Error_PC/v3/data.pkl","rb"))
cmap = matplotlib.cm.get_cmap('viridis')
normalize_to_tendon_opt_slack_ratio = matplotlib.colors.Normalize(
    vmin=1/max(stuff['Slack_Opt_Tendon_Ratio']),
    vmax=1/min(stuff['Slack_Opt_Tendon_Ratio'])
)
half_dlT = np.zeros(len(stuff['cT'][:1000]))
for i in range(len(stuff['cT'][:1000])):
    l_To_over_l_Ts = 1/stuff['Slack_Opt_Tendon_Ratio'][i]
    lT = np.linspace(1,l_To_over_l_Ts,1001)
    cT = stuff['cT'][i]
    kT = stuff['kT'][i]
    fT = cT*kT*np.log(np.exp(((1/l_To_over_l_Ts)*lT - 1 + 1/cT)/kT)+1)
    half_dlT[i] = (
        l_To_over_l_Ts
        * kT
        * np.log(
            (np.exp(0.5/(cT*kT))-1)
            / (np.exp(stuff['initial_force']/(cT*kT))-1)
        )
    )
    plt.plot(lT,fT,c=cmap(normalize_to_tendon_opt_slack_ratio(l_To_over_l_Ts)))
ax=plt.gca()
ax.set_xlim([1,1.08])
ax.set_xticks([1,1.01,1.02,1.03,1.04,1.05,1.06,1.07])
ax.set_xticklabels(["1","","","1.03","","","","1.07"])
ax.set_ylim([0,1.05])
ax.set_yticks([0,0.5,1])
ax.set_yticklabels([r"$0\%$",r"$50\%$",r"$100\%$"])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(
    [
        1/max(stuff['Slack_Opt_Tendon_Ratio']),
        1/max(stuff['Slack_Opt_Tendon_Ratio'])
    ],
    [0,1],
    'k--'
)
plt.plot([0,1/min(stuff['Slack_Opt_Tendon_Ratio'])],[0.5,0.5],'k--')
plt.plot(
    [
        0,
        1/min(stuff['Slack_Opt_Tendon_Ratio']),
        1/min(stuff['Slack_Opt_Tendon_Ratio'])
    ],
    [1,1,0],
    'k--'
)
ax.set_xlabel(r"$\hat{l}^T$",fontsize=14)
ax.set_ylabel(r"$\hat{f}^T$",fontsize=14)
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap)
cax.set_ylabel(r"$\dfrac{l_{T,o}}{l_{T,s}}$",fontsize=14)
cbar.set_ticks(
    [
        (
            int(
                100*(el-np.floor(1/(max(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01)
                / (
                    np.ceil(1/(min(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01
                    - np.floor(1/(max(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01
                )
            )/100
        )
        for el in np.arange(
            np.floor(1/(max(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01,
            np.ceil(1/(min(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01+1e-4,
            0.01
        )
    ]
)
cax.set_yticklabels(
    [
        '{:0.2f}'.format(el)
        for el in np.arange(
            np.floor(1/(max(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01,
            np.ceil(1/(min(stuff['Slack_Opt_Tendon_Ratio']))/0.01)*0.01+1e-4,
            0.01
        )
    ]
)
plt.show()
