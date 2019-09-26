import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np
from math import acos
from scipy.optimize import fsolve
#
# class MidpointNormalize(colors.Normalize):
# 	"""
# 	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
#
# 	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
# 	"""
# 	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
# 		self.midpoint = midpoint
# 		colors.Normalize.__init__(self, vmin, vmax, clip)
#
# 	def __call__(self, value, clip=None):
# 		# I'm ignoring masked values and all kinds of edge cases to make a
# 		# simple example...
# 		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
# 		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
#

cmap = matplotlib.cm.get_cmap('viridis')

### Coefficient 1 contour plot

fig_1 = plt.figure(figsize=(10,10))
ax_11 = fig_1.add_subplot(2,2,(1,3))
ax_12 = fig_1.add_subplot(2,2,2)
ax_13 = fig_1.add_subplot(2,2,4)

pc = np.linspace(0,40*np.pi/180,1001)
p = np.linspace(0,40*np.pi/180,1001)
P,PC = np.meshgrid(p,pc)
C1 = (np.cos(PC)-np.cos(P))/(np.cos(PC)*np.cos(P))
CM1 = ax_11.contour(
    180*P/np.pi,
    180*PC/np.pi,
    100*C1,
    list(np.arange(-35,36,5))
)

fmt = {}
bounds = 35
strs = ["-35%","-30%","-25%","-20%","-15%","-10%","-5%",
    "Zero Error",
    "5%","10%","15%","20%","25%","30%","35%"
]
for l, s in zip(CM1.levels, strs):
    fmt[l] = s

coords_0 = []
percent_0 = np.linspace(-0.3,0.3,7)
for i in range(len(percent_0)):
    p_temp = fsolve(
        lambda p: (
            np.cos(40*np.pi/180-p)
            - np.cos(p)
            - percent_0[i]*np.cos(p)*np.cos(40*np.pi/180-p)
        ),
        0
    )
    coords_0.append((p_temp[0]*180/np.pi,40 - p_temp[0]*180/np.pi))

# Label every other level using strings
ax_11.clabel(
    CM1,
    CM1.levels[1::2],
    inline=True,
    fmt=fmt,
    fontsize=10,
    manual=coords_0
)
ax_11.set_xlabel(r"$\rho$" + " (deg.)")
ax_11.set_ylabel(r"$\rho_c$" + " (deg.)")

normalize = matplotlib.colors.Normalize(
    vmin=-bounds,
    vmax=bounds
)

high = 30
high_C1 = (
    (np.cos(high*np.pi/180)-np.cos(p))
    / (np.cos(high*np.pi/180)*np.cos(p))
)
colors_list_high = [cmap(normalize(value)) for value in 100*high_C1]
ax_11.plot([0,40],[high,high],'k',lw=3)
ax_12.scatter(
    180*p/np.pi,
    100*(
        (np.cos(high*np.pi/180)-np.cos(p))
        / (np.cos(high*np.pi/180)*np.cos(p))
    ),
    c=colors_list_high,
    marker="."
)
ax_12.set_ylim([-35,35])
ax_12.set_xlim([0,40])
ax_12.spines['right'].set_visible(False)
ax_12.spines['top'].set_visible(False)
ax_12.spines['bottom'].set_position('zero')
ax_12.set_ylabel(r'$C_1$' + ' (%)')
ax_12.set_title("High " + r"$\rho_c$")

low = 10
low_C1 = (
    (np.cos(low*np.pi/180)-np.cos(p))
    / (np.cos(low*np.pi/180)*np.cos(p))
)
colors_list_low = [cmap(normalize(value)) for value in 100*low_C1]
ax_11.plot([0,40],[low,low],'k',lw=3)
ax_13.scatter(
    180*p/np.pi,
    100*(
        (np.cos(low*np.pi/180)-np.cos(p))
        / (np.cos(low*np.pi/180)*np.cos(p))
    ),
    c=colors_list_low,
    marker="."
)
ax_13.set_ylim([-35,35])
ax_13.set_xlim([0,40])
ax_13.spines['right'].set_visible(False)
ax_13.spines['top'].set_visible(False)
ax_13.spines['bottom'].set_position('zero')
ax_13.set_ylabel(r'$C_1$' + ' (%)')
ax_13.set_xlabel(r'$\rho$' + ' (deg.)')
ax_13.set_title("Low " + r"$\rho_c$")



### Coefficient 2 contour plot

fig_2 = plt.figure(figsize=(10,10))
ax_21 = fig_2.add_subplot(2,2,(1,3))
ax_22 = fig_2.add_subplot(2,2,2)
ax_23 = fig_2.add_subplot(2,2,4)

po = np.linspace(0,40*np.pi/180,1001)
p = np.linspace(0,40*np.pi/180,1001)
P,PO = np.meshgrid(p,pc)
C2 = (np.cos(PO)-np.cos(P))/np.cos(P)
CM2 = ax_21.contour(
    180*P/np.pi,
    180*PO/np.pi,
    100*C2,
    list(np.arange(-35,36,5))
)

fmt = {}
bounds = 35
strs = ["-35%","-30%","-25%","-20%","-15%","-10%","-5%",
    "Zero Error",
    "5%","10%","15%","20%","25%","30%","35%"
]
for l, s in zip(CM2.levels, strs):
    fmt[l] = s

coords_1 = []
percent_1 = np.linspace(-0.3,0.3,7)
for i in range(len(percent_1)):
    p_temp = fsolve(
        lambda p: (
            np.cos(40*np.pi/180-p)
            - np.cos(p)
            - percent_1[i]*np.cos(p)
        ),
        0
    )
    coords_1.append((p_temp[0]*180/np.pi,40 - p_temp[0]*180/np.pi))

# Label every other level using strings
ax_21.clabel(
    CM2,
    CM2.levels[1::2],
    inline=True,
    fmt=fmt,
    fontsize=10,
    manual=coords_1
)
ax_21.set_xlabel(r"$\rho$" + " (deg.)")
ax_21.set_ylabel(r"$\rho(t_o)$" + " (deg.)")

normalize = matplotlib.colors.Normalize(
    vmin=-bounds,
    vmax=bounds
)

high = 30
high_C1 = (
    (np.cos(high*np.pi/180)-np.cos(p))
    / np.cos(p)
)
colors_list_high = [cmap(normalize(value)) for value in 100*high_C1]
ax_21.plot([0,40],[high,high],'k',lw=3)
ax_22.scatter(
    180*p/np.pi,
    100*(
        (np.cos(high*np.pi/180)-np.cos(p))
        / np.cos(p)
    ),
    c=colors_list_high,
    marker="."
)
ax_22.set_ylim([-25,35])
ax_22.set_xlim([0,40])
ax_22.spines['right'].set_visible(False)
ax_22.spines['top'].set_visible(False)
ax_22.spines['bottom'].set_position('zero')
ax_22.set_ylabel(r'$C_2$' + ' (%)')
ax_22.set_title("High " + r"$\rho(t_o)$")

low = 10
low_C1 = (
    (np.cos(low*np.pi/180)-np.cos(p))
    / np.cos(p)
)
colors_list_low = [cmap(normalize(value)) for value in 100*low_C1]
ax_21.plot([0,40],[low,low],'k',lw=3)
ax_23.scatter(
    180*p/np.pi,
    100*(
        (np.cos(low*np.pi/180)-np.cos(p))
        / np.cos(p)
    ),
    c=colors_list_low,
    marker="."
)
ax_23.set_ylim([-25,35])
ax_23.set_xlim([0,40])
ax_23.spines['right'].set_visible(False)
ax_23.spines['top'].set_visible(False)
ax_23.spines['bottom'].set_position('zero')
ax_23.set_ylabel(r'$C_2$' + ' (%)')
ax_23.set_xlabel(r'$\rho$' + ' (deg.)')
ax_23.set_title("Low " + r"$\rho(t_o)$")



### C1 variability for fixed delta

pc = np.linspace(0,40*np.pi/180,1001)
delta = 20*np.pi/180
C1_upper = (np.cos(pc)-np.cos(pc+delta))/(np.cos(pc)*np.cos(pc+delta))
C1_lower = (np.cos(pc)-np.cos(pc-delta))/(np.cos(pc)*np.cos(pc-delta))
for i in range(len(pc)):
    if pc[i]-delta<0:
        C1_lower[i] = (np.cos(pc[i])-np.cos(0))/(np.cos(pc[i])*np.cos(0))

fig_3 = plt.figure()
ax_3 = plt.gca()
ax_3.plot(pc,100*C1_upper,'0.70')
ax_3.plot(pc,100*C1_lower,'0.70')
ax_3.set_title("Coefficient Variability Range\n" + r"for $\rho\in[\rho_c-\delta\rho,\rho_c+\delta\rho]$")
ax_3.set_xlabel("Pennation (deg.)")
ax_3.set_ylabel("Coefficient (%)")
ax_3.set_xticks([0,np.pi/18,2*np.pi/18,3*np.pi/18,4*np.pi/18])
ax_3.set_xticklabels(["0","10","20","30","40"])


### C1 Sensitivity

pc = np.linspace(0,40*np.pi/180,9)
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(pc), vmax=max(pc))
colors_list = [cmap(normalize(value)) for value in pc]
fig_4 = plt.figure()
ax_4 = plt.gca()
ax_4.set_title("Coefficient Variability \n" + r"as $\rho_c$ Changes")
ax_4.set_xlabel("Pennation (deg.)")
ax_4.set_ylabel("Coefficient (%)")
ax_4.set_xticks([0,np.pi/18,2*np.pi/18,3*np.pi/18,4*np.pi/18])
ax_4.set_xticklabels(["","","","","40"])
p = np.linspace(0,50*np.pi/180,1001)
ax_4.spines['bottom'].set_position('zero')
ax_4.spines['right'].set_visible(False)
ax_4.spines['top'].set_visible(False)
ax_4.set_xlim([0,50*np.pi/180])
for i in range(len(pc)):
    delta = 5*np.pi/180
    if pc[i]-delta>0:
        p = np.linspace(pc[i]-delta,pc[i]+delta,1001)
    else:
        p = np.linspace(0,pc[i]+delta,1001)
    C1 = (np.cos(pc[i])-np.cos(p))/(np.cos(pc[i])*np.cos(p))
    plt.plot(p,100*C1,c=colors_list[i])
cax_4, _ = matplotlib.colorbar.make_axes(ax_4)
cbar_4 = matplotlib.colorbar.ColorbarBase(cax_4, cmap=cmap)
cax_4.set_ylabel(r"$\rho_c$" + " (deg.)",fontsize=14)
cbar_4.set_ticks([0,0.25,0.5,0.75,1.0])
cax_4.set_yticklabels(["0","10","20","30","40"])

p = np.linspace(0,40*np.pi/180,1001)
p_10 = acos(1/1.1)
fig_5 = plt.figure()
ax_5 = plt.gca()
ax_5.plot(p,100*(1-np.cos(p))/np.cos(p),c=colors_list[0])
ax_5.set_title(r"Assuming $\rho(t)\approx 0$")
ax_5.set_xlabel("Pennation (deg.)")
ax_5.set_ylabel("Coefficient (%)")
ax_5.spines['right'].set_visible(False)
ax_5.spines['top'].set_visible(False)
ax_5.set_xticks([0,np.pi/18,2*np.pi/18,3*np.pi/18,4*np.pi/18])
ax_5.set_xticklabels(["0","10","20","30","40"])
ax_5.plot([0,p_10,p_10],[10,10,0],'k--')
ax_5.set_xlim([0,40*np.pi/180])
ax_5.set_ylim([0,100*(1-np.cos(p[-1]))/np.cos(p[-1])])

po = np.linspace(0,40*np.pi/180,9)
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(po), vmax=max(po))
colors_list = [cmap(normalize(value)) for value in pc]
fig_6 = plt.figure()
ax_6 = plt.gca()
ax_6.spines['bottom'].set_position('zero')
ax_6.spines['right'].set_visible(False)
ax_6.spines['top'].set_visible(False)
ax_6.set_title("Coefficient Variability \n" + r"as $\rho(t_o)$ Changes")
ax_6.set_xlabel("Pennation (deg.)")
ax_6.set_ylabel("Coefficient (%)")
ax_6.set_xticks([0,np.pi/18,2*np.pi/18,3*np.pi/18,4*np.pi/18])
ax_6.set_xticklabels(["0","10","20","30","40"])
for i in range(len(po)):
    delta = 5*np.pi/180
    if pc[i]-delta>0:
        p = np.linspace(pc[i]-delta,pc[i]+delta,1001)
    else:
        p = np.linspace(0,pc[i]+delta,1001)
    actual = (np.cos(po[i])-np.cos(p))/(np.cos(p))
    test = (p**2-po[i]**2)/(2-p**2)
    plt.plot(p,100*actual,c=colors_list[i])
cax_6, _ = matplotlib.colorbar.make_axes(ax_6)
cbar_6 = matplotlib.colorbar.ColorbarBase(cax_6, cmap=cmap)
cax_6.set_ylabel(r"$\rho(t_o)$" + " (deg.)",fontsize=14)
cbar_6.set_ticks([0,0.25,0.5,0.75,1.0])
cax_6.set_yticklabels(["0","10","20","30","40"])
# ax2.plot([p[0],p[-1]],[0,0],'k--')

plt.show()
