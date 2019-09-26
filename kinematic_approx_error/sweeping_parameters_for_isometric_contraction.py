import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from scipy.optimize import fsolve
import ipdb
from danpy.useful_functions import save_figures
from danpy.sb import dsb

#########################################
#### Finding the bounding conditions ####
#########################################

l_Ts_over_l_To = 0.95
initial_force = 0.001
long_cT_array = np.linspace(10,100,10001)
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
kT_array = np.linspace(0,kT_intersection,101)

lb_cT = np.zeros(np.shape(kT_array))
ub_cT = np.zeros(np.shape(kT_array))
for i in range(len(kT_array)):
    if (long_cT_array*kT_array[i]<0.20).all():
        ub_cT[i] = 100
    else:
        ub_cT[i] = 0.20/kT_array[i]
    if kT_array[i]==0:
        lb_cT[i] = (1-initial_force)/(1-l_Ts_over_l_To)
    else:
        lT_initial = (1/l_Ts_over_l_To)*(
            kT_array[i]*np.log(
                np.exp(initial_force/(long_cT_array*kT_array[i]))
                - 1
            )
            + (1-1/long_cT_array)
        )
        lb_cT[i] = long_cT_array[int(sum(lT_initial<1))-1]

cT_outline = np.concatenate(
    [
        lb_cT,
        list(reversed(ub_cT)),
        [lb_cT[0]]
    ]
)
kT_outline = np.concatenate(
    [
        np.linspace(0,kT_intersection,101),
        list(reversed(np.linspace(0,kT_intersection,101))),
        [0]
    ]
)

#########################################
###### Create cT, kT, and fT arrays #####
#########################################

cT_array = np.linspace(10,100,31)
kT_array = np.linspace(0,kT_intersection,31)
F_LB = 0
F_UB = 1

CT,KT = np.meshgrid(cT_array,kT_array)
fT_array = np.linspace(F_LB,F_UB,11)[1:]


#########################################
######## Find delta_lT for X% MVC #######
#########################################

# for submaximal activations since fT = 1 when lT = 1/l_Ts_over_l_To
delta_lT_max = (1/l_Ts_over_l_To) - 1

cmap = matplotlib.cm.get_cmap('viridis')
normalize_to_delta_lT_max = matplotlib.colors.Normalize(
    vmin=0,
    vmax=delta_lT_max
)
normalize_to_lT_max = matplotlib.colors.Normalize(
    vmin=1,
    vmax=1+delta_lT_max
)

fig1 = plt.figure(figsize=(15,6))
ax1 = Axes3D(fig1)
ax1.set_title("All Levels")
ax1.set_xlim([15,105])
ax1.set_xticks([20,40,60,80,100])
ax1.set_xlabel(r"$c^T \propto$" + " Asymptotic Stiffness")
ax1.set_ylim([0,1])
ax1.set_yticks([0,0.20,0.40,0.60,0.80,1])
ax1.set_yticklabels(["0%","20%","40%","60%","80%","100%"])
ax1.set_ylabel("Isometric Tendon Force (%" + r"$F_{MAX}$"+")")
ax1.set_zlim([0,np.ceil(kT_intersection/0.0005)/2000])
ax1.set_zticks(
    np.arange(
        0,
        np.round(kT_intersection/0.001)*0.001 + 1e-8,
        0.002
    )
)
ax1.set_zlabel(r"$k^T \propto$" +" Radius of Curvature")
ax1.view_init(elev=35., azim=-30.)
cax1, _ = matplotlib.colorbar.make_axes(ax1)
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=cmap)
cax1.set_ylabel(r"$\Delta\hat{l}_T$",fontsize=14)
cbar1.set_ticks(
    [
        el/delta_lT_max
        for el in np.linspace(
            0,
            np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)
cax1.set_yticklabels(
    [
        '{:0.2f}'.format(el)
        for el in np.linspace(
            0,
            np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)

statusbar = dsb(0,len(fT_array),title="Sweeping % MVC")
for i in range(len(fT_array)):
    fig2 = plt.figure(figsize=(15,6))
    ax2 = Axes3D(fig2)

    FT = fT_array[i]*np.ones(np.shape(CT))
    for j in range(len(cT_array)-1):
        for k in range(len(kT_array)-1):
            kT = np.average([KT[k,j],KT[k+1,j]])
            cT = np.average([CT[k,j],CT[k,j+1]])
            delta_lT = kT*np.log(
                (np.exp(fT_array[i]/(cT*kT))-1)
                /((np.exp(initial_force/(cT*kT))-1))
            )
            verts = [
                list(
                    zip(
                        [CT[k,j],CT[k,j+1],CT[k+1,j+1],CT[k+1,j]],
                        [FT[k,j],FT[k,j+1],FT[k+1,j+1],FT[k+1,j]],
                        [KT[k,j],KT[k,j+1],KT[k+1,j+1],KT[k+1,j]]
                    )
                )
            ]
            ax1.add_collection3d(
                Poly3DCollection(
                    verts,
                    facecolor=cmap(normalize_to_delta_lT_max(delta_lT)),
                    alpha=0.75
                )
            )
            ax2.add_collection3d(
                Poly3DCollection(
                    verts,
                    facecolor=cmap(normalize_to_delta_lT_max(delta_lT)),
                    alpha=0.75
                )
            )

    ax2.plot(
        cT_outline,
        fT_array[i]*np.ones(np.shape(cT_outline)),
        kT_outline,
        c='k'
    )
    ax2.set_xlim([15,105])
    ax2.set_xticks([20,40,60,80,100])
    ax2.set_xlabel(r"$c^T \propto$" + " Asymptotic Stiffness")
    ax2.set_ylim([0,1])
    ax2.set_yticks([0,0.20,0.40,0.60,0.80,1])
    ax2.set_yticklabels(["0%","20%","40%","60%","80%","100%"])
    ax2.set_ylabel("Isometric Tendon Force (%" + r"$F_{MAX}$"+")")
    ax2.set_zlim([0,np.ceil(kT_intersection/0.0005)/2000]) # kt_max = 0.0062
    ax2.set_zticks(
        np.arange(
            0,
            np.round(kT_intersection/0.001)*0.001 + 1e-8,
            0.002
        )
    )
    ax2.set_zlabel(r"$k^T \propto$" +" Radius of Curvature")
    ax2.view_init(elev=35., azim=-30.)
    cax2, _ = matplotlib.colorbar.make_axes(ax2)
    cbar2 = matplotlib.colorbar.ColorbarBase(cax2, cmap=cmap)
    cax2.set_ylabel(r"$\Delta\hat{l}_T$",fontsize=14)
    cbar2.set_ticks(
        [
            el/delta_lT_max
            for el in np.linspace(
                0,
                np.round(delta_lT_max/0.01)*0.01,
                int(np.round(delta_lT_max/0.01)) + 1
            )
        ]
    )
    cax2.set_yticklabels(
        [
            '{:0.2f}'.format(el)
            for el in np.linspace(
                0,
                np.round(delta_lT_max/0.01)*0.01,
                int(np.round(delta_lT_max/0.01)) + 1
            )
        ]
    )

    if i==0:
        Path = save_figures(
            "./Figures/Iso_Error/",
            "Iso_"+str(100*fT_array[i])+"_MVC",
            {
                "Initial Force" : initial_force,
                "Slack to Optimal Ratio" : l_Ts_over_l_To
            },
            figs=[fig2],
            SaveAsPDF=True,
            ReturnPath=True
        )
    else:
        save_figures(
            "./Figures/Iso_Error/",
            "Iso_"+str(int(100*fT_array[i]))+"_MVC",
            {},
            SubFolder=Path[-18:],
            figs=[fig2],
            SaveAsPDF=True
        )
    plt.close(fig2)
    statusbar.update(i)

save_figures(
    "./Figures/Iso_Error/",
    "Iso_MVC_all_levels",
    {},
    SubFolder=Path[-18:],
    figs=[fig1],
    SaveAsPDF=True
)
plt.close(fig1)

# #########################################
# ###### Find delta_lT for All levels #####
# #########################################
#
# statusbar = dsb(0,len(fT_array),title="All Levels")
# for i in range(len(fT_array)):
#     FT = fT_array[i]*np.ones(np.shape(CT))
#     for j in range(len(cT_array)-1):
#         for k in range(len(kT_array)-1):
#             kT = np.average([KT[k,j],KT[k+1,j]])
#             cT = np.average([CT[k,j],CT[k,j+1]])
#             lT_initial = (1/l_Ts_over_l_To)*(
#                 kT*np.log(
#                     np.exp(initial_force/(cT*kT))
#                     - 1
#                 )
#                 + (1-1/cT)
#             )
#             delta_lT = kT*np.log(
#                 (np.exp(fT_array[i]/(cT*kT))-1)
#                 /((np.exp(initial_force/(cT*kT))-1))
#             )
#             verts = [
#                 list(
#                     zip(
#                         [CT[k,j],CT[k,j+1],CT[k+1,j+1],CT[k+1,j]],
#                         [FT[k,j],FT[k,j+1],FT[k+1,j+1],FT[k+1,j]],
#                         [KT[k,j],KT[k,j+1],KT[k+1,j+1],KT[k+1,j]]
#                     )
#                 )
#             ]
#             ax2.add_collection3d(
#                 Poly3DCollection(
#                     verts,
#                     facecolor=cmap(normalize_to_delta_lT_max(delta_lT)),
#                     alpha=0.75
#                 )
#             )
#     statusbar.update(i)
#

#########################################
######### delta_lT for 100% MVC #########
#########################################

fig3 = plt.figure(figsize=(12,3))
ax31 = fig3.add_subplot(1,3,1)
ax32 = fig3.add_subplot(1,3,2)
ax33 = fig3.add_subplot(1,3,3)

for i in range(len(kT_array)-1):
    for j in range(len(cT_array)-1):
        kT = np.average([KT[i,j],KT[i+1,j]])
        cT = np.average([CT[i,j],CT[i,j+1]])
        lT_initial = (1/l_Ts_over_l_To)*(
            kT*np.log(
                np.exp(initial_force/(cT*kT))
                - 1
            )
            + (1-1/cT)
        )
        lT_final = (1/l_Ts_over_l_To)*(
            kT*np.log(
                np.exp(1/(cT*kT))
                - 1
            )
            + (1-1/cT)
        )
        delta_lT = kT*np.log(
            (np.exp(1/(cT*kT))-1)
            /((np.exp(initial_force/(cT*kT))-1))
        )
        verts = np.array(
            [
                [CT[i,j],KT[i,j]],
                [CT[i,j+1],KT[i,j]],
                [CT[i,j+1],KT[i+1,j]],
                [CT[i,j],KT[i+1,j]]
            ]
        )
        ax31.add_patch(
            matplotlib.patches.Polygon(
                verts,
                fc=cmap(normalize_to_lT_max(lT_initial)),
                alpha=0.75
            )
        )
        ax32.add_patch(
            matplotlib.patches.Polygon(
                verts,
                fc=cmap(normalize_to_lT_max(lT_final)),
                alpha=0.75
            )
        )
        ax33.add_patch(
            matplotlib.patches.Polygon(
                verts,
                fc=cmap(normalize_to_delta_lT_max(delta_lT)),
                alpha=0.75
            )
        )

ax31.plot(
    cT_outline,
    kT_outline,
    c='k'
)
ax32.plot(
    cT_outline,
    kT_outline,
    c='k'
)
ax33.plot(
    cT_outline,
    kT_outline,
    c='k'
)
ax31.set_title("Initial Tendon Length")
ax31.set_ylabel(r"$k^T$")
ax31.set_ylim([0,np.ceil(kT_intersection/0.0005)/2000])
ax31.set_yticks(
    np.arange(
        0,
        np.round(kT_intersection/0.001)*0.001 + 1e-8,
        0.002
    )
)
ax31.set_xlabel(r"$c^T$")
ax31.set_xlim([(initial_force-1)/(l_Ts_over_l_To-1),100])
ax31.set_xticks(
    np.linspace(
        20*np.ceil((initial_force-1)/(l_Ts_over_l_To-1)/20),
        cT_array[-1],
        (
            int(np.ceil(cT_array[-1]/20))
            - int(np.ceil((initial_force-1)/(l_Ts_over_l_To-1)/20))
            + 1
        )
    )
)

cax31, _ = matplotlib.colorbar.make_axes(ax31)
cbar31 = matplotlib.colorbar.ColorbarBase(cax31, cmap=cmap)
cax31.set_ylabel(r"$\hat{l}_T(t_o)$",fontsize=14)
cbar31.set_ticks(
    [
        (el-1)/delta_lT_max
        for el in np.linspace(
            1,
            1 + np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)
cax31.set_yticklabels(
    [
        '{:0.2f}'.format(el)
        for el in np.linspace(
            1,
            1 + np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)


ax32.set_title("Final Tendon Length")
ax32.set_ylabel(r"$k^T$")
ax32.set_ylim([0,np.ceil(kT_intersection/0.0005)/2000])
ax32.set_yticks(
    np.arange(
        0,
        np.round(kT_intersection/0.001)*0.001 + 1e-8,
        0.002
    )
)
# ax32.set_yticklabels([""]*len(ax32.get_yticks()))
ax32.set_xlabel(r"$c^T$")
ax32.set_xlim([(initial_force-1)/(l_Ts_over_l_To-1),100])
ax32.set_xticks(
    np.linspace(
        20*np.ceil((initial_force-1)/(l_Ts_over_l_To-1)/20),
        cT_array[-1],
        (
            int(np.ceil(cT_array[-1]/20))
            - int(np.ceil((initial_force-1)/(l_Ts_over_l_To-1)/20))
            + 1
        )
    )
)
# ax32.set_xticklabels([""]*len(ax32.get_xticks()))

cax32, _ = matplotlib.colorbar.make_axes(ax32)
cbar32 = matplotlib.colorbar.ColorbarBase(cax32, cmap=cmap)
cax32.set_ylabel(r"$\hat{l}_T(t_f)$",fontsize=14)
cbar32.set_ticks(
    [
        (el-1)/delta_lT_max
        for el in np.linspace(
            1,
            1 + np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)
cax32.set_yticklabels(
    [
        '{:0.2f}'.format(el)
        for el in np.linspace(
            1,
            1+np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)

ax33.set_title("Change in Tendon Length")
ax33.set_ylabel(r"$k^T$")
ax33.set_ylim([0,np.ceil(kT_intersection/0.0005)/2000])
ax33.set_yticks(
    np.arange(
        0,
        np.round(kT_intersection/0.001)*0.001 + 1e-8,
        0.002
    )
)
# ax33.set_yticklabels([""]*len(ax33.get_yticks()))
ax33.set_xlabel(r"$c^T$")
ax33.set_xlim([(initial_force-1)/(l_Ts_over_l_To-1),100])
ax33.set_xticks(
    np.linspace(
        20*np.ceil((initial_force-1)/(l_Ts_over_l_To-1)/20),
        cT_array[-1],
        (
            int(np.ceil(cT_array[-1]/20))
            - int(np.ceil((initial_force-1)/(l_Ts_over_l_To-1)/20))
            + 1
        )
    )
)
# ax33.set_xticklabels([""]*len(ax33.get_xticks()))

cax33, _ = matplotlib.colorbar.make_axes(ax33)
cbar33 = matplotlib.colorbar.ColorbarBase(cax33, cmap=cmap)
cax33.set_ylabel(r"$\Delta\hat{l}_T$",fontsize=14)
cbar33.set_ticks(
    [
        el/delta_lT_max
        for el in np.linspace(
            0,
            np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)
cax33.set_yticklabels(
    [
        '{:0.2f}'.format(el)
        for el in np.linspace(
            0,
            np.round(delta_lT_max/0.01)*0.01,
            int(np.round(delta_lT_max/0.01)) + 1
        )
    ]
)

save_figures(
    "./Figures/Iso_Error/",
    "Iso_100_MVC_initial_final_comparison",
    {},
    SubFolder=Path[-18:],
    SaveAsPDF=True
)

plt.close(fig3)
