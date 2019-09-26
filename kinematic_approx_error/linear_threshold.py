import numpy as np
import matplotlib
import matplotlib.pyplot as plt


alpha = 0.05
lin_strain = 1.02
l_Ts_over_l_To = 0.95/1
l_To_over_l_Ts = 1/l_Ts_over_l_To
cT_array = np.linspace(10,100,200)
kT_max = 0.2/min(cT_array)
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors_list.Normalize(vmin=min(cT_array), vmax=max(cT_array))
colors_list = [cmap(normalize(value)) for value in cT_array]
max_thresh=0
min_kT=np.zeros(len(cT_array))
max_kT=0
for i in range(len(cT_array)):
    # kT_min = (1 - (1/cT_array[i]) - l_Ts_over_l_To*lin_strain)/np.log(alpha/(1-alpha))
    # if kT_min < 0:
    #     kT_min = 0
    # if kT_min > 0.2/cT_array[i]:
    #     kT_min = 0.2/cT_array[i]
    kT_array = np.linspace(0,0.2/cT_array[i],1001)
    thresh = l_To_over_l_Ts*(1-(1/cT_array[i])-kT_array*np.log(alpha/(1-alpha)))
    plt.plot(kT_array,thresh,c=colors_list[i])
    max_thresh = max([max_thresh,max(thresh)])
    max_kT = max([max_kT, max(kT_array)])
    min_kT[i] = kT_array[np.where(0.5*(thresh-1)**2==min(0.5*(thresh-1)**2))]
ax = plt.gca()
ax.set_xticks(list(np.arange(0,0.0201,0.005)))
ax.set_ylim([lin_strain-0.05*(max_thresh-lin_strain),max_thresh+0.05*(max_thresh-lin_strain)])
ax.set_xlim([-0.05*max_kT,1.05*max_kT])
ax.set_xlabel(r"$k^T$",fontsize=14)
ax.set_ylabel(r"$\hat{l}_{T,thresh}$",fontsize=14)
cax1, _ = matplotlib.colorbar.make_axes(ax)
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=cmap)
cax1.set_ylabel(r"$c^T$",fontsize=14)
cbar1.set_ticks([0,1/9,2/9,3/9,4/9,5/9,6/9,7/9,8/9,9/9])
cax1.set_yticklabels(["10","20","30","40","50","60","70","80","90","100"]);


### Plotting the linear threshold as a function of cT and kT
### Note that for cT within normal values, these thresholds are within 1.5-4%
alpha=0.05
kT_max = 0.2/min(cT_array)
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors_list.Normalize(vmin=0, vmax=kT_max)
for i in range(len(cT_array)):
    kT_array = np.linspace(0,0.20/cT_array[i],101)
    colors_list = [cmap(normalize(value)) for value in kT_array]
    linear_threshold = l_To_over_l_Ts*(1-1/cT_array[i]-kT_array*np.log(alpha/(1-alpha)))
    plt.scatter([cT_array[i]]*len(kT_array),linear_threshold,c=colors_list)
plt.plot(cT_array, l_To_over_l_Ts*(1-(1/cT_array)*(1+0.20*np.log(alpha/(1-alpha)))),'k--')
ax = plt.gca()
ax.plot([30,30],[0,2],'k--')
ax.plot([80,80],[0,2],'k--')
ax.set_ylim([0.95,1.1])
ax.set_xlabel(r"$c^T$",fontsize=14)
ax.set_ylabel(r"$\hat{l}_{T,thresh}$",fontsize=14)
cax1, _ = matplotlib.colorbar.make_axes(ax)
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=cmap)
cax1.set_ylabel(r"$k^T$",fontsize=14)
cbar1.set_ticks([0,1])
cax1.set_yticklabels(["0","0.02"])


### Plotting the effect of sweepting kT on the FL plot
cT = 25
lT = np.linspace(0.98,1/0.95,1001)
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors_list.Normalize(vmin=min(kT), vmax=max(kT))
colors_list = [cmap(normalize(value)) for value in kT]
for i in range(len(kT)):
    plt.plot(lT,cT*kT[i]*np.log(np.exp((l_Ts_to_l_To_ratio*lT-1+1/cT)/kT[i])+1),c=colors_list[i])
ax = plt.gca()
cax1, _ = matplotlib.colorbar.make_axes(ax)
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap=cmap)
cbar1.set_ticks([0,1])
cax1.set_yticklabels([r"$k^T_{\min}$",r"$k^T_{\max}$"])
######
