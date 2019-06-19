##################################################
### Replicating the Plots from Kawakami (1998) ###
##################################################

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from danpy.sb import dsb

F_MAX_MG = 1000
F_MAX_LG = 1000
F_MAX_Sol = 1000
cT = 27.8
kT = 0.0047
lTo_MG = 0.45 # in m
lTo_LG = 0.45 # in m
lTo_Sol = 0.45 # in m

R = 0.05 # m

AnkleAngles = np.zeros((4,3))
KneeAngles = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        AnkleAngles[i,j] = i*15 - 15
        KneeAngles[i,j] = j*45

PennationAngle_Passive_MG = np.array([
    [22*np.pi/180,26*np.pi/180,34*np.pi/180],
    [24*np.pi/180,29*np.pi/180,39*np.pi/180],
    [27*np.pi/180,34*np.pi/180,42*np.pi/180],
    [31*np.pi/180,38*np.pi/180,45*np.pi/180]
]) # in radians

PennationAngle_Passive_LG = np.array([
    [12*np.pi/180,13*np.pi/180,12*np.pi/180],
    [13*np.pi/180,14*np.pi/180,14*np.pi/180],
    [15*np.pi/180,15*np.pi/180,16*np.pi/180],
    [16*np.pi/180,16*np.pi/180,17*np.pi/180]
]) # in radians

PennationAngle_Passive_Sol = np.array([
    [19*np.pi/180,19*np.pi/180,19*np.pi/180],
    [21*np.pi/180,22*np.pi/180,21*np.pi/180],
    [25*np.pi/180,25*np.pi/180,24*np.pi/180],
    [29*np.pi/180,29*np.pi/180,28*np.pi/180]
]) # in radians

PennationAngle_MVC_MG = np.array([
    [33*np.pi/180,44*np.pi/180,54*np.pi/180],
    [40*np.pi/180,51*np.pi/180,62*np.pi/180],
    [46*np.pi/180,55*np.pi/180,65*np.pi/180],
    [48*np.pi/180,58*np.pi/180,67*np.pi/180]
]) # in radians

PennationAngle_MVC_LG = np.array([
    [19*np.pi/180,21*np.pi/180,25*np.pi/180],
    [24*np.pi/180,25*np.pi/180,29*np.pi/180],
    [28*np.pi/180,29*np.pi/180,31*np.pi/180],
    [31*np.pi/180,34*np.pi/180,35*np.pi/180]
]) # in radians

PennationAngle_MVC_Sol = np.array([
    [33*np.pi/180,33*np.pi/180,33*np.pi/180],
    [40*np.pi/180,39*np.pi/180,40*np.pi/180],
    [45*np.pi/180,45*np.pi/180,45*np.pi/180],
    [49*np.pi/180,49*np.pi/180,49*np.pi/180]
]) # in radians

MuscleLength_Passive_MG = np.array([
    [59/1000,47/1000,38/1000],
    [52/1000,42/1000,35/1000],
    [45/1000,38/1000,33/1000],
    [40/1000,35/1000,32/1000]
]) # in m

MuscleLength_Passive_LG = np.array([
    [65/1000,59/1000,53/1000],
    [56/1000,51/1000,46/1000],
    [51/1000,46/1000,42/1000],
    [47/1000,43/1000,41/1000]
]) # in m

MuscleLength_Passive_Sol = np.array([
    [43/1000,43/1000,43/1000],
    [38/1000,39/1000,38/1000],
    [33/1000,34/1000,33/1000],
    [29/1000,29/1000,30/1000]
]) # in m

MuscleLength_MVC_MG = np.array([
    [38/1000,30/1000,27/1000],
    [31/1000,26/1000,26/1000],
    [28/1000,26/1000,26/1000],
    [26/1000,25/1000,25/1000]
]) # in m

MuscleLength_MVC_LG = np.array([
    [46/1000,40/1000,35/1000],
    [38/1000,34/1000,31/1000],
    [33/1000,31/1000,29/1000],
    [30/1000,28/1000,27/1000]
]) # in m

MuscleLength_MVC_Sol = np.array([
    [31/1000,31/1000,32/1000],
    [26/1000,26/1000,26/1000],
    [24/1000,24/1000,23/1000],
    [23/1000,22/1000,23/1000]
]) # in m

ChangeInLength_MG = np.zeros((4,3))
ChangeInLength_LG = np.zeros((4,3))
ChangeInLength_Sol = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        ChangeInLength_MG[i,j] = (
            np.cos(PennationAngle_MVC_MG[i,j])
            * MuscleLength_MVC_MG[i,j]
            - np.cos(PennationAngle_Passive_MG[i,j])
            * MuscleLength_Passive_MG[i,j]
        )
        ChangeInLength_LG[i,j] = (
            np.cos(PennationAngle_MVC_LG[i,j])
            * MuscleLength_MVC_LG[i,j]
            - np.cos(PennationAngle_Passive_LG[i,j])
            * MuscleLength_Passive_LG[i,j]
        )
        ChangeInLength_Sol[i,j] = (
            np.cos(PennationAngle_MVC_Sol[i,j])
            * MuscleLength_MVC_Sol[i,j]
            - np.cos(PennationAngle_Passive_Sol[i,j])
            * MuscleLength_Passive_Sol[i,j]
        )

# InitialTension_MG = np.array([
#     [80,80,80],
#     [80,80,80],
#     [80,80,80],
#     [80,80,80]
# ]) # in N
# InitialTension_LG = np.array([
#     [80,80,80],
#     [80,80,80],
#     [80,80,80],
#     [80,80,80]
# ]) # in N
# InitialTension_Sol = np.array([
#     [80,80,80],
#     [80,80,80],
#     [80,80,80],
#     [80,80,80]
# ]) # in N
InitialTension_MG = (MuscleLength_Passive_MG-MuscleLength_Passive_MG.min())*1000 # in N
InitialTension_LG = (MuscleLength_Passive_LG-MuscleLength_Passive_LG.min())*1000 # in N
InitialTension_Sol = (MuscleLength_Passive_Sol-MuscleLength_Passive_Sol.min())*1000 # in N

Tension_MG = np.zeros((4,3))
Tension_LG = np.zeros((4,3))
Tension_Sol = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        Tension_MG[i,j] = (
            F_MAX_MG*cT*kT*np.log(
                np.exp(-ChangeInLength_MG[i,j]/(kT*lTo_MG))
                * (np.exp(InitialTension_MG[i,j]/(F_MAX_MG*cT*kT))-1)
                + 1
            )
        )
        Tension_LG[i,j] = (
            F_MAX_LG*cT*kT*np.log(
                np.exp(-ChangeInLength_LG[i,j]/(kT*lTo_LG))
                * (np.exp(InitialTension_LG[i,j]/(F_MAX_LG*cT*kT))-1)
                + 1
            )
        )
        Tension_Sol[i,j] = (
            F_MAX_Sol*cT*kT*np.log(
                np.exp(-ChangeInLength_Sol[i,j]/(kT*lTo_Sol))
                * (np.exp(InitialTension_Sol[i,j]/(F_MAX_Sol*cT*kT))-1)
                + 1
            )
        )

Torque_MG = Tension_MG*R #Scalar mult will not change shape but to test magnitude
Torque_LG = Tension_LG*R #Scalar mult will not change shape but to test magnitude
Torque_Sol = Tension_Sol*R #Scalar mult will not change shape but to test magnitude

x = AnkleAngles.flatten() # in deg
y1 = -ChangeInLength_MG.flatten()*1000 # in mm
y2 = -ChangeInLength_LG.flatten()*1000 # in mm
y3 = -ChangeInLength_Sol.flatten()*1000 # in mm
y4 = (y1+y2+y3)/3 # in mm
z1 = Torque_MG.flatten() # in Nm
z2 = Torque_LG.flatten() # in Nm
z3 = Torque_Sol.flatten() # in Nm
z4 = (z1+z2+z3) # in Nm

marker = [
    's',"D",'o',
    's',"D",'o',
    's',"D",'o',
    's',"D",'o'
]

markerfacecolor = [
    'k','w','k',
    'k','w','k',
    'k','w','k',
    'k','w','k'
]

fig1 = plt.figure(figsize=[15,5])
ax1a = fig1.add_subplot(131, projection='3d')
for i in range(len(x)):
    ax1a.scatter(x[i], y1[i], z1[i], c='k', marker=marker[i])
    ax1a.plot(
        [x[i],x[i]],
        [y1[i],y1[i]],
        [0,z1[i]],
        c='k'
    )
ax1a.set_title("Medial Gastrocnemius \n")
ax1a.set_xlabel('Ankle Angle (deg)')
ax1a.set_ylabel('Change in Muscle Length (mm)')
ax1a.set_zlabel('Torque (Nm)')

ax1b = fig1.add_subplot(132, projection='3d')
for i in range(len(x)):
    ax1b.scatter(x[i], y2[i], z2[i], c='k', marker=marker[i])
    ax1b.plot(
        [x[i],x[i]],
        [y2[i],y2[i]],
        [0,z2[i]],
        c='k'
    )
ax1b.set_title("Lateral Gastrocnemius \n")
ax1b.set_xlabel('Ankle Angle (deg)')
ax1b.set_ylabel('Change in Muscle Length (mm)')
ax1b.set_zlabel('Torque (Nm)')

ax1c = fig1.add_subplot(133, projection='3d')
for i in range(len(x)):
    ax1c.scatter(x[i], y3[i], z3[i], c='k', marker=marker[i])
    ax1c.plot(
        [x[i],x[i]],
        [y3[i],y3[i]],
        [0,z3[i]],
        c='k'
    )
ax1c.set_title("Soleus \n")
ax1c.set_xlabel('Ankle Angle (deg)')
ax1c.set_ylabel('Change in Muscle Length (mm)')
ax1c.set_zlabel('Torque (Nm)')

fig2 = plt.figure(figsize=[11,5])
ax2a = fig2.add_subplot(121)
img0 = mpimg.imread("useful_figures/Average_muscle_length_change_vs_angle_vs_torque.png")
ax2a.imshow(img0)
ax2a.spines['right'].set_visible(False)
ax2a.spines['top'].set_visible(False)
ax2a.spines['left'].set_visible(False)
ax2a.spines['bottom'].set_visible(False)
ax2a.set_xticks([])
ax2a.set_yticks([])
ax2b = fig2.add_subplot(122, projection='3d')
for i in range(len(x)):
    ax2b.scatter(x[i], y4[i], z4[i], c='k', marker=marker[i])
    ax2b.plot(
        [x[i],x[i]],
        [y4[i],y4[i]],
        [0,z4[i]],
        c='k'
    )
fig2.suptitle("Average Muscle")
ax2b.set_xlabel('Ankle Angle (deg)')
ax2b.set_ylabel('Change in Muscle Length (mm)')
ax2b.set_zlabel('Torque (Nm)')
ax2b.set_xticks([-15,0,15,30])
ax2b.set_yticks([12,14,16,18,20])
ax2b.set_zticks([0,40,80,120,160,200])
ax2b.set_xlim([-15,30])
ax2b.set_ylim([12,20])
ax2b.set_zlim([0,200])

fig3 = plt.figure(figsize=[15,9])
ax3a = fig3.add_subplot(231)
for i in range(12):
    ax3a.plot(
        (MuscleLength_Passive_MG.flatten()*1000)[i],
        y1[i],
        c='k',
        marker=marker[i],
        markerfacecolor=markerfacecolor[i]
    )
ax3a.set_xlim([25,65])
ax3a.set_ylim([10,25])
ax3a.set_xticks([25,35,45,55,65])
ax3a.set_yticks([10,15,20,25])
ax3a.spines['right'].set_visible(False)
ax3a.spines['top'].set_visible(False)
ax3a.set_xlabel('Passive Muscle Length (mm)')
ax3a.set_ylabel('Muscle Length Change (mm)')
ax3a.set_title("Medial Gastrocnemius \n (Simulated)")

ax3b = fig3.add_subplot(232)
for i in range(12):
    plt.plot(
        (MuscleLength_Passive_LG.flatten()*1000)[i],
        y2[i],
        c='k',
        marker=marker[i],
        markerfacecolor=markerfacecolor[i]
    )
ax3b.set_xlim([25,65])
ax3b.set_ylim([10,25])
ax3b.set_xticks([25,35,45,55,65])
ax3b.set_yticks([10,15,20,25])
ax3b.spines['right'].set_visible(False)
ax3b.spines['top'].set_visible(False)
ax3b.set_xlabel('Passive Muscle Length (mm)')
ax3b.set_ylabel('Muscle Length Change (mm)')
ax3b.set_title("Lateral Gastrocnemius \n (Simulated)")

ax3c = fig3.add_subplot(233)
for i in range(12):
    plt.plot(
        (MuscleLength_Passive_Sol.flatten()*1000)[i],
        y3[i],
        c='k',
        marker=marker[i],
        markerfacecolor=markerfacecolor[i]
    )
ax3c.set_xlim([25,65])
ax3c.set_ylim([10,25])
ax3c.set_xticks([25,35,45,55,65])
ax3c.set_yticks([10,15,20,25])
ax3c.spines['right'].set_visible(False)
ax3c.spines['top'].set_visible(False)
ax3c.set_xlabel('Passive Muscle Length (mm)')
ax3c.set_ylabel('Muscle Length Change (mm)')
ax3c.set_title("Soleus \n (Simulated)")

ax3d = fig3.add_subplot(234)
ax3d.set_title("Kurokawa (2008)")
img1 = mpimg.imread('useful_figures/MG_length_change_vs_passive_length.png')
ax3d.imshow(img1)
ax3d.spines['right'].set_visible(False)
ax3d.spines['top'].set_visible(False)
ax3d.spines['left'].set_visible(False)
ax3d.spines['bottom'].set_visible(False)
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_xlabel('Passive Muscle Length (mm)')
ax3d.set_ylabel('Muscle Length Change (mm)')

ax3e = fig3.add_subplot(235)
ax3e.set_title("Kurokawa (2008)")
img2 = mpimg.imread('useful_figures/LG_length_change_vs_passive_length.png')
ax3e.imshow(img2)
ax3e.spines['right'].set_visible(False)
ax3e.spines['top'].set_visible(False)
ax3e.spines['left'].set_visible(False)
ax3e.spines['bottom'].set_visible(False)
ax3e.set_xticks([])
ax3e.set_yticks([])
ax3e.set_xlabel('Passive Muscle Length (mm)')
ax3e.set_ylabel('Muscle Length Change (mm)')

ax3f = fig3.add_subplot(236)
ax3f.set_title("Kurokawa (2008)")
img3 = mpimg.imread('useful_figures/Sol_length_change_vs_passive_length.png')
ax3f.imshow(img3)
ax3f.spines['right'].set_visible(False)
ax3f.spines['top'].set_visible(False)
ax3f.spines['left'].set_visible(False)
ax3f.spines['bottom'].set_visible(False)
ax3f.set_xticks([])
ax3f.set_yticks([])
ax3f.set_xlabel('Passive Muscle Length (mm)')
ax3f.set_ylabel('Muscle Length Change (mm)')

plt.show()
