import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy import signal
from math import atan2
from danpy.sb import dsb

ls0 = 1
ls1 = 2
ls2 = 2
x5o = (ls0**2 + ls2**2 - ls1**2)/(2*ls0)
x7o = - np.sqrt(ls2**2-x5o**2)
x9o = (ls0**2 + ls2**2 - ls1**2)/(2*ls0)
x11o = np.sqrt(ls2**2-x9o**2)
k0 = 0.1
k1 = 100
k2 = 100

def Delta_l0(X):
    return(np.sqrt((X[0]+ls0)**2 + X[2]**2) - ls0)

def Delta_l11(X):
    return(
        np.sqrt(
            (X[0]-X[4]+ls0-x5o)**2
            + (X[2]-X[6]-x7o)**2
        )
        - np.sqrt(
            (ls0-x5o)**2
            + (x7o)**2
        )
    )
def Delta_l12(X):
    return(
        np.sqrt(
            (X[0]-X[8]+ls0-x9o)**2
            + (X[2]-X[10]-x11o)**2
        )
        - np.sqrt(
            (ls0-x9o)**2
            + (x11o)**2
        )
    )
def Delta_l21(X):
    return(
        np.sqrt((X[4]+x5o)**2 + (X[6]+x7o)**2)
        - np.sqrt(x5o**2 + x7o**2)
    )
def Delta_l22(X):
    return(
        np.sqrt((X[8]+x9o)**2 + (X[10]+x11o)**2)
        - np.sqrt(x9o**2 + x11o**2)
    )

def dX1(X,F):
    return(X[1])
def dX2(X,F):
    return(
        F[0]
        - k0*Delta_l0(X)*(X[0]+ls0)/np.sqrt((X[0]+ls0)**2 + X[2]**2)
        - (
            k1
            * Delta_l11(X)
            * (X[0]-X[4]+ls0-x5o)
            / np.sqrt(
                (X[0]-X[4]+ls0-x5o)**2
                + (X[2]-X[6]-x7o)**2
            )
        )
        - (
            k1
            * Delta_l12(X)
            * (X[0]-X[8]+ls0-x9o)
            / np.sqrt(
                (X[0]-X[8]+ls0-x9o)**2
                + (X[2]-X[10]-x11o)**2
            )
        )
    )
def dX3(X,F):
    return(X[3])
def dX4(X,F):
    return(
        F[1]
        - k0*Delta_l0(X)*(X[2])/np.sqrt((X[0]+ls0)**2 + X[2]**2)
        - (
            k1
            * Delta_l11(X)
            * (X[2]-X[6]-x7o)
            / np.sqrt(
                (X[0]-X[4]+ls0-x5o)**2
                + (X[2]-X[6]-x7o)**2
            )
        )
        - (
            k1
            * Delta_l12(X)
            * (X[2]-X[10]-x11o)
            / np.sqrt(
                (X[0]-X[8]+ls0-x9o)**2
                + (X[2]-X[10]-x11o)**2
            )
        )
    )
def dX5(X,F):
    return(X[5])
def dX6(X,F):
    return(
        - (
            k2
            * Delta_l21(X)
            * (X[4]+x5o)
            / np.sqrt((X[4]+x5o)**2 + (X[6]+x7o)**2)
        )
        + (
            k1
            * Delta_l11(X)
            * (X[0]-X[4]+ls0-x5o)
            / np.sqrt(
                (X[0]-X[4]+ls0-x5o)**2
                + (X[2]-X[6]-x7o)**2
            )
        )
    )
def dX7(X,F):
    return(X[7])
def dX8(X,F):
    return(
        - (
            k2
            * Delta_l21(X)
            * (X[6]+x7o)
            / np.sqrt((X[4]+x5o)**2 + (X[6]+x7o)**2)
        )
        + (
            k1
            * Delta_l11(X)
            * (X[2]-X[6]-x7o)
            / np.sqrt(
                (X[0]-X[4]+ls0-x5o)**2
                + (X[2]-X[6]-x7o)**2
            )
        )
    )
def dX9(X,F):
    return(X[9])
def dX10(X,F):
    return(
        - (
            k2
            * Delta_l22(X)
            * (X[8]+x9o)
            / np.sqrt((X[8]+x9o)**2 + (X[10]+x11o)**2)
        )
        + (
            k1
            * Delta_l12(X)
            * (X[0]-X[8]+ls0-x9o)
            / np.sqrt(
                (X[0]-X[8]+ls0-x9o)**2
                + (X[2]-X[10]-x11o)**2
            )
        )
    )
def dX11(X,F):
    return(X[11])
def dX12(X,F):
    return(
        - (
            k2
            * Delta_l22(X)
            * (X[10]+x11o)
            / np.sqrt((X[8]+x9o)**2 + (X[10]+x11o)**2)
        )
        + (
            k1
            * Delta_l12(X)
            * (X[2]-X[10]-x11o)
            / np.sqrt(
                (X[0]-X[8]+ls0-x9o)**2
                + (X[2]-X[10]-x11o)**2
            )
        )
    )

dt = 0.001
Time = np.arange(0,20+dt,dt)

X = np.zeros((12,len(Time)))
X[:,0] = [
    0,0,
    0,0,
    0,0,
    0,0,
    0,0,
    0,0
]

F_levels = np.linspace(0,100,21)
Averages = np.zeros((1,len(F_levels)))
statusbar = dsb(0,len(F_levels),title="Testing Crimped Tendon")
for j in range(len(F_levels)):
    F_level = F_levels[j]
    tau = 2
    F = np.zeros((2,len(Time)))
    # F[0,:] = F_level*np.ones((1,len(Time)))
    F[0,:] = F_level*(1-np.exp(-Time/tau))**2
    for i in range(len(Time)-1):
        X[0,i+1] = X[0,i] + dX1(X[:,i],F[:,i])*dt
        X[1,i+1] = X[1,i] + dX2(X[:,i],F[:,i])*dt
        X[2,i+1] = X[2,i] + dX3(X[:,i],F[:,i])*dt
        X[3,i+1] = X[3,i] + dX4(X[:,i],F[:,i])*dt
        X[4,i+1] = X[4,i] + dX5(X[:,i],F[:,i])*dt
        X[5,i+1] = X[5,i] + dX6(X[:,i],F[:,i])*dt
        X[6,i+1] = X[6,i] + dX7(X[:,i],F[:,i])*dt
        X[7,i+1] = X[7,i] + dX8(X[:,i],F[:,i])*dt
        X[8,i+1] = X[8,i] + dX9(X[:,i],F[:,i])*dt
        X[9,i+1] = X[9,i] + dX10(X[:,i],F[:,i])*dt
        X[10,i+1] = X[10,i] + dX11(X[:,i],F[:,i])*dt
        X[11,i+1] = X[11,i] + dX12(X[:,i],F[:,i])*dt
    Averages[0,j] = np.average(X[0,-int(2/dt):])
    statusbar.update(j)
plt.scatter(Averages,F_levels,c='k',marker='*')

def animate_trajectory(response,Time,x1,x3,x4,u1,u2):
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot2grid((3,4),(0,0),colspan=4)
    ax2 = plt.subplot2grid((3,4),(1,0),colspan=2)
    ax3 = plt.subplot2grid((3,4),(1,2),colspan=2)
    ax4 = plt.subplot2grid((3,4),(2,0),colspan=3)
    ax5 = plt.subplot2grid((3,4),(2,3))

    plt.suptitle("Underdetermined Mass-Spring System",Fontsize=28,y=0.95)

    # Model Drawing
    IdealBoxScalingFactor = 0.78533496170320571 # Calculated from w = np.pi
    CurrentTrialScalingFactor = max([max(x3)-min(x1),max(x1)-min(x4)])

    RestingLength = max([max(x1)-min(x3),max(x4)-min(x1)])+2*StraightLength\
                    +0.30*CurrentTrialScalingFactor/IdealBoxScalingFactor
    CenterBoxHalfWidth = 0.15*CurrentTrialScalingFactor/IdealBoxScalingFactor
    CenterBoxHalfHeight = 0.2*CurrentTrialScalingFactor/IdealBoxScalingFactor
    SideBoxHalfWidth = 0.1*CurrentTrialScalingFactor/IdealBoxScalingFactor
    SideBoxHalfHeight = 0.075*CurrentTrialScalingFactor/IdealBoxScalingFactor
    ForceScaling = 1*CurrentTrialScalingFactor/IdealBoxScalingFactor

    StraightLength0 = 0.25*ls0
    Spring_array0 =(
        (0.05*ls0)
        * np.abs(signal.sawtooth(5*2*np.pi*np.linspace(0,1,1001)-np.pi/2))\
        - (1/2)*(0.05*ls0)
    )

    Spring0, = ax1.plot(
        np.linspace(StraightLength0,ls0+X[0,0]-StraightLength0,1001),
        Spring_array0,
        'k'
    )
    Spring0_left, = ax1.plot(
        [0,StraightLength0],
        [0,0],
        'k'
    )
    Spring1_right, = ax1.plot(
        [ls0+X[0,0]-StraightLength0,ls0+X[0,0]],
        [0,0],
        'k'
    )

    StraightLength1 = 0.25*ls1
    alpha1 = atan2(X[6,0]+x7o-X[2,0],X[4,0]+x5o-X[0,0]-ls0)
    Spring_array1 = (
        (0.05*ls1)
        * np.abs(signal.sawtooth(5*2*np.pi*np.linspace(0,1,1001)-np.pi/2))
        - (1/2)*(0.05*ls1)
    )[:,np.newaxis]
    Spring_array1 = np.concatenate(
        [
            np.linspace(
                StraightLength1,
                np.sqrt(
                    (X[4,0]+x5o-X[0,0]-ls0)**2
                    + (X[6,0]+x7o-X[2,0])**2
                )-StraightLength1,
                1001
            )[:,np.newaxis],
            Spring_array1
        ],
        axis=1
    ).T
    Spring_total1 = (
        np.matrix([
            [np.cos(alpha1),-np.sin(alpha1)],
            [np.sin(alpha1),np.cos(alpha1)]
        ]) * Spring_array1
        + np.concatenate(
            [
                ls0+X[0,0]*np.ones((1,1001)),
                X[1,0]*np.ones((1,1001))
            ],
            axis=0
        )
    )
    Spring1, = ax1.plot(
        np.array(Spring_total1[0,:])[0],
        np.array(Spring_total1[1,:])[0],
        'k'
    )
    Spring1_left, = ax1.plot(
        [ls0+X[0,0],StraightLength1*np.cos(alpha1)+ls0+X[0,0]],
        [X[2,0],StraightLength1*np.sin(alpha1)+X[2,0]],
        'k'
    )
    Spring1_right, = ax1.plot(
        [
            (
                np.sqrt(
                    (X[4,0]+x5o-X[0,0]-ls0)**2
                    + (X[6,0]+x7o-X[2,0])**2
                )
                - StraightLength1
            ) * np.cos(alpha1) + ls0 + X[0,0],
            np.sqrt(
                (X[4,0]+x5o-X[0,0]-ls0)**2
                + (X[6,0]+x7o-X[2,0])**2
            )*np.cos(alpha1) + ls0 + X[0,0]
        ],
        [
            (
                np.sqrt(
                    (X[4,0]+x5o-X[0,0]-ls0)**2
                    + (X[6,0]+x7o-X[2,0])**2
                )
                - StraightLength1
            ) * np.sin(alpha1) + X[2,0],
            np.sqrt(
                (X[4,0]+x5o-X[0,0]-ls0)**2
                + (X[6,0]+x7o-X[2,0])**2
            )*np.sin(alpha1) + X[2,0]
        ],
        'k'
    )

    StraightLength2 = 0.25*ls2
    alpha2 = atan2(X[6,0]+x7o,X[4,0]+x5o)
    Spring_array2 = (
        (0.05*ls2)
        * np.abs(signal.sawtooth(5*2*np.pi*np.linspace(0,1,1001)-np.pi/2))
        - (1/2)*(0.05*ls2)
    )[:,np.newaxis]
    Spring_array2 = np.concatenate(
        [
            np.linspace(
                StraightLength2,
                np.sqrt((X[4,0]+x5o)**2 + (X[6,0]+x7o)**2)-StraightLength2,
                1001
            )[:,np.newaxis],
            Spring_array2
        ],
        axis=1
    ).T
    Spring_total2 = (
        np.matrix([
            [np.cos(alpha2),-np.sin(alpha2)],
            [np.sin(alpha2),np.cos(alpha2)]
        ]) * Spring_array2
    )
    Spring2, = ax1.plot(
        np.array(Spring_total2[0,:])[0],
        np.array(Spring_total2[1,:])[0],
        'k'
    )
    Spring2_left, = ax1.plot(
        [0,StraightLength2*np.cos(alpha2)],
        [0,StraightLength2*np.sin(alpha2)],
        'k'
    )
    Spring2_right, = ax1.plot(
        [
            (
                np.sqrt((X[4,0]+x5o)**2 + (X[6,0]+x7o)**2)
                - StraightLength2
            ) * np.cos(alpha2),
            np.sqrt((X[4,0]+x5o)**2 + (X[6,0]+x7o)**2)*np.cos(alpha2)
        ],
        [
            (
                np.sqrt((X[4,0]+x5o)**2 + (X[6,0]+x7o)**2)
                - StraightLength2
            ) * np.sin(alpha2),
            np.sqrt((X[4,0]+x5o)**2 + (X[6,0]+x7o)**2)*np.sin(alpha2)
        ],
        'k'
    )

    Spring2_right, = \
        ax1.plot([-RestingLength+x4[0]+SideBoxHalfWidth,\
                    -RestingLength+x4[0]+SideBoxHalfWidth+StraightLength],\
                        [0,0],'k')

    ############### LEFT OF HERE IDIOT....
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.set_frame_on(True)
    CenterMass = plt.Rectangle((-CenterBoxHalfWidth,-CenterBoxHalfHeight),\
                                2*CenterBoxHalfWidth,2*CenterBoxHalfHeight,Color='#4682b4')
    ax1.add_patch(CenterMass)
    Mass1 = plt.Rectangle((-SideBoxHalfWidth+RestingLength,-SideBoxHalfHeight),\
                                2*SideBoxHalfWidth,2*SideBoxHalfHeight,Color='#4682b4')
    ax1.add_patch(Mass1)
    Mass2 = plt.Rectangle((-SideBoxHalfWidth-RestingLength,-SideBoxHalfHeight),\
                                2*SideBoxHalfWidth,2*SideBoxHalfHeight,Color='#4682b4')
    ax1.add_patch(Mass2)

    PositionArrow, = ax1.plot([x1[0],x1[0]],[0,2*CenterBoxHalfHeight],'k')
    PositionArrowHead, = ax1.plot([x1[0]],[2*CenterBoxHalfHeight],'k^')
    PositionArrowTail, = ax1.plot([x1[0]],[0],'ko')

    Scale = ax1.plot([-1.1*A,1.1*A],\
                        [2.75*CenterBoxHalfHeight,2.75*CenterBoxHalfHeight],\
                            '0.60')
    Ticks = np.linspace(-A,A,5)
    TickHeights = [0.3*CenterBoxHalfHeight,\
                    0.15*CenterBoxHalfHeight,\
                    0.3*CenterBoxHalfHeight,\
                    0.15*CenterBoxHalfHeight,\
                    0.3*CenterBoxHalfHeight]
    [ax1.plot([Ticks[i],Ticks[i]],\
            [2.75*CenterBoxHalfHeight-TickHeights[i],2.75*CenterBoxHalfHeight],'0.60') \
                for i in range(5)]

    Force1Arrow, = ax1.plot([RestingLength+x3[0]+(5/3)*SideBoxHalfWidth,\
                                RestingLength + x3[0]+(5/3)*SideBoxHalfWidth\
                                    +ForceScaling*u1[0]/(max(u1[5000:]+u2[5000:]))],\
                                        [0,0],'g')
    Force1ArrowHead, = \
        ax1.plot([RestingLength + x3[0]+(5/3)*SideBoxHalfWidth\
                    +ForceScaling*u1[0]/(max(u1[5000:]+u2[5000:]))],[0],'g>')
    Force2Arrow, =\
        ax1.plot([x4[0]-RestingLength-(5/3)*SideBoxHalfWidth\
                    -ForceScaling*u2[0]/(max(u1[5000:]+u2[5000:])),\
                        x4[0]-RestingLength-(5/3)*SideBoxHalfWidth],[0,0],'r')
    Force2ArrowHead, = \
        ax1.plot([x4[0]-RestingLength-(5/3)*SideBoxHalfWidth\
                    -ForceScaling*u2[0]/(max(u1[5000:]+u2[5000:]))],[0],'r<')

    LowerBound = (np.array(x4[5001:])-RestingLength-(5/3)*SideBoxHalfWidth\
                    -ForceScaling*np.array(u2[5000:])/(max(u1[5000:]+u2[5000:]))).min()
    UpperBound = (RestingLength + np.array(x3[5001:])+(5/3)*SideBoxHalfWidth\
                    +ForceScaling*np.array(u1[5000:])/(max(u1[5000:]+u2[5000:]))).max()
    Bound = 1.05*np.array([-LowerBound,UpperBound]).max()
    ax1.set_xlim([-Bound,Bound])
    ax1.set_ylim([-1.5*CenterBoxHalfHeight,3.25*CenterBoxHalfHeight])
    ax1.set_aspect('equal')

    #Force 1

    Force1, = ax3.plot([0],[u1[0]],color = 'g')
    ax3.set_xlim(0,Time[-1])
    ax3.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax3.set_xticklabels([str(0),'','','',str(Time[-1])])
    ax3.set_ylim(0,1.15*max(u1[5000:]+u2[5000:]))
    if np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                    int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1).shape[0] < 5:
        ax3.set_yticks(list(np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                        int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1)))
        ax3.set_yticklabels([""]*(int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1))
    else:
        NumTicks = np.floor(1.15*max(u1[5000:]+u2[5000:]))
        MaxTick = NumTicks - NumTicks%5
        TickStep = MaxTick/5
        Ticks = list(np.linspace(0,TickStep*5,6))
        ax3.set_yticks(Ticks)
        ax3.set_yticklabels([""]*len(Ticks))
    # ax3.set_yticklabels([str(int(el)) for el in \
    #                         list(np.linspace(0,\
    #                             np.ceil(max(u1[int(len(u1)/2):])*1.1) - \
    #                                 np.ceil(max(u1[int(len(u1)/2):])*1.1)%3,4))],\
    #                                     fontsize=12)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_title("Force 1",fontsize=16,fontweight = 4,color = 'g',y = 0.95)
    # ax3.set_xlabel("Time (s)")

    #Force 2

    Force2, = ax2.plot([0],[u2[0]],color = 'r')
    ax2.set_xlim(0,Time[-1])
    ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
    ax2.set_ylim(0,1.15*max(u1[5000:]+u2[5000:]))
    ax2.set_yticks(list(np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                    int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1)))
    ax2.set_yticklabels([str(int(el)) for el in \
                            list(np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                                int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1))],\
                                    fontsize=12)
    if np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                    int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1).shape[0] < 5:
        ax2.set_yticks(list(np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                        int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1)))
        ax2.set_yticklabels([str(int(el)) for el in \
                                list(np.linspace(0,np.floor(1.15*max(u1[5000:]+u2[5000:])),\
                                    int(np.floor(1.15*max(u1[5000:]+u2[5000:])))+1))],\
                                        fontsize=12)
    else:
        NumTicks = np.floor(1.15*max(u1[5000:]+u2[5000:]))
        MaxTick = NumTicks - NumTicks%5
        TickStep = MaxTick/5
        Ticks = list(np.linspace(0,TickStep*5,6))
        ax2.set_yticks(Ticks)
        ax2.set_yticklabels([str(tick) for tick in Ticks])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title("Force 2",fontsize=16,fontweight = 4,color = 'r',y = 0.95)
    # ax2.set_xlabel("Time (s)")

    # Trajectory

    Predicted, = ax4.plot(Time,r(Time),'0.60',linestyle='--')
    Actual, = ax4.plot([0],[x1[0]],'b')
    ax4.set_xlim(0,Time[-1])
    ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
    ax4.set_ylim([-1.25*A,1.25*A])
    ax4.set_yticks([-A,0,A])
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Position of Center Mass (m)")
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)

    # Error
    ErrorArray = x1-r(Time)
    Error, = ax5.plot([0],[ErrorArray[0]],'k')
    ax5.set_xlim(0,Time[-1])
    ax5.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax5.set_xticklabels([str(0),'','','',str(Time[-1])])
    ax5.set_ylim([ErrorArray.min() - 0.1*(max(ErrorArray)-min(ErrorArray)),\
                    ErrorArray.max() + 0.1*(max(ErrorArray)-min(ErrorArray))])
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Error (m)")
    ax5.yaxis.set_label_position("right")
    ax5.yaxis.tick_right()
    ax5.spines['left'].set_visible(False)
    ax5.spines['top'].set_visible(False)

    def animate(i):
        Spring1.set_xdata(np.linspace(x1[i]+CenterBoxHalfWidth+StraightLength,\
                                RestingLength+x3[i]-SideBoxHalfWidth-StraightLength,1001))
        Spring1_left.set_xdata([x1[i]+CenterBoxHalfWidth,x1[i]+CenterBoxHalfWidth+StraightLength])
        Spring1_right.set_xdata([RestingLength+x3[i]-SideBoxHalfWidth-StraightLength,\
                                    RestingLength+x3[i]-SideBoxHalfWidth])

        Spring2.set_xdata(np.linspace(-RestingLength+x4[i]+SideBoxHalfWidth+StraightLength,\
                                x1[i]-CenterBoxHalfWidth-StraightLength,1001))
        Spring2_left.set_xdata([x1[i]-CenterBoxHalfWidth-StraightLength,x1[i]-CenterBoxHalfWidth])
        Spring2_right.set_xdata([-RestingLength+x4[i]+SideBoxHalfWidth,\
                                    -RestingLength+x4[i]+SideBoxHalfWidth+StraightLength])

        CenterMass.xy = (-CenterBoxHalfWidth + x1[i],-CenterBoxHalfHeight)
        Mass1.xy = (-SideBoxHalfWidth+RestingLength + x3[i],-SideBoxHalfHeight)
        Mass2.xy = (-SideBoxHalfWidth-RestingLength + x4[i],-SideBoxHalfHeight)
        PositionArrow.set_xdata([x1[i],x1[i]])
        PositionArrowHead.set_xdata([x1[i]])
        PositionArrowTail.set_xdata([x1[i]])
        Force1Arrow.set_xdata([RestingLength+x3[i]+(5/3)*SideBoxHalfWidth,\
                                RestingLength + x3[i]+(5/3)*SideBoxHalfWidth\
                                    +ForceScaling*u1[i]/(max(u1[5000:]+u2[5000:]))])
        Force1ArrowHead.set_xdata([RestingLength + x3[i]+(5/3)*SideBoxHalfWidth\
                                    +ForceScaling*u1[i]/(max(u1[5000:]+u2[5000:]))])
        Force2Arrow.set_xdata([x4[i]-RestingLength-(5/3)*SideBoxHalfWidth\
                                    -ForceScaling*u2[i]/(max(u1[5000:]+u2[5000:])),\
                                        x4[i]-RestingLength-(5/3)*SideBoxHalfWidth])
        Force2ArrowHead.set_xdata([x4[i]-RestingLength-(5/3)*SideBoxHalfWidth\
                                    -ForceScaling*u2[i]/(max(u1[5000:]+u2[5000:]))])

        Force1.set_xdata(Time[:i])
        Force1.set_ydata(u1[:i])

        Force2.set_xdata(Time[:i])
        Force2.set_ydata(u2[:i])

        Actual.set_xdata(Time[:i])
        Actual.set_ydata(x1[:i])

        Error.set_xdata(Time[:i])
        Error.set_ydata(ErrorArray[:i])

        return Spring1,Spring1_left,Spring1_right,Spring2,Spring2_left,Spring2_right,CenterMass,Mass1,Mass2,Force1,Force2,Actual,Error,PositionArrow,PositionArrowHead,PositionArrowTail,Force1Arrow,Force1ArrowHead,Force2Arrow,Force2ArrowHead,

    # Init only required for blitting to give a clean slate.
    def init():
        Spring1, =\
            ax1.plot(np.linspace(x1[0]+CenterBoxHalfWidth+StraightLength,\
                                    RestingLength+x3[0]-SideBoxHalfWidth-StraightLength,1001),\
                                        Spring_array,'k')
        Spring1_left, = \
            ax1.plot([x1[0]+CenterBoxHalfWidth,x1[0]+CenterBoxHalfWidth+StraightLength],\
                [0,0],'k')
        Spring1_right, = \
            ax1.plot([RestingLength+x3[0]-SideBoxHalfWidth-StraightLength,\
                        RestingLength+x3[0]-SideBoxHalfWidth],[0,0],'k')
        Spring2, =\
            ax1.plot(np.linspace(-RestingLength+x4[0]+SideBoxHalfWidth+StraightLength,\
                                    x1[0]-CenterBoxHalfWidth-StraightLength,1001),\
                                        Spring_array,'k')
        Spring2_left, =\
            ax1.plot([x1[0]-CenterBoxHalfWidth-StraightLength,x1[0]-CenterBoxHalfWidth],\
                        [0,0],'k')
        Spring2_right, = \
            ax1.plot([-RestingLength+x4[0]+SideBoxHalfWidth,\
                        -RestingLength+x4[0]+SideBoxHalfWidth+StraightLength],[0,0],'k')

        CenterMass = plt.Rectangle((-CenterBoxHalfWidth,-CenterBoxHalfHeight),\
                                    2*CenterBoxHalfWidth,2*CenterBoxHalfHeight,Color='#4682b4')
        ax1.add_patch(CenterMass)
        Mass1 = plt.Rectangle((-SideBoxHalfWidth+RestingLength,-SideBoxHalfHeight),\
                                    2*SideBoxHalfWidth,2*SideBoxHalfHeight,Color='#4682b4')
        ax1.add_patch(Mass1)
        Mass2 = plt.Rectangle((-SideBoxHalfWidth-RestingLength,-SideBoxHalfHeight),\
                                    2*SideBoxHalfWidth,2*SideBoxHalfHeight,Color='#4682b4')
        ax1.add_patch(Mass2)

        PositionArrow, = ax1.plot([x1[0],x1[0]],[0,2*CenterBoxHalfHeight],'k')
        PositionArrowHead, = ax1.plot([x1[0]],[2*CenterBoxHalfHeight],'k^')
        PositionArrowTail, = ax1.plot([x1[0]],[0],'ko')

        Force1Arrow, = ax1.plot([RestingLength+x3[0]+(5/3)*SideBoxHalfWidth,\
                                RestingLength + x3[0]+(5/3)*SideBoxHalfWidth\
                                    +ForceScaling*u1[0]/(max(u1[5000:]+u2[5000:]))],\
                                        [0,0],'g')
        Force1ArrowHead, = \
            ax1.plot([RestingLength + x3[0]+(5/3)*SideBoxHalfWidth\
                    +ForceScaling*u1[0]/(max(u1[5000:]+u2[5000:]))],[0],'g<')
        Force2Arrow, = ax1.plot([x4[0]-RestingLength-(5/3)*SideBoxHalfWidth\
                    -ForceScaling*u2[0]/(max(u1[5000:]+u2[5000:])),\
                        x4[0]-RestingLength-(5/3)*SideBoxHalfWidth],[0,0],'r')
        Force2ArrowHead, = \
            ax1.plot([x4[0]-RestingLength-(5/3)*SideBoxHalfWidth\
                -ForceScaling*u2[0]/(max(u1[5000:]+u2[5000:]))],[0],'r>')

        Force1, = ax3.plot([0],[u1[0]],color = 'g')
        Force2, = ax2.plot([0],[u2[0]],color = 'r')
        Predicted, = ax4.plot(Time,r(Time),'0.60',linestyle='--')
        Actual, = ax4.plot([0],[x1[0]],'b')
        Error, = ax5.plot([0],[ErrorArray[0]],'k')

        Spring1.set_visible(False)
        Spring1_left.set_visible(False)
        Spring1_right.set_visible(False)
        Spring2.set_visible(False)
        Spring2_left.set_visible(False)
        Spring2_right.set_visible(False)
        CenterMass.set_visible(False)
        Mass1.set_visible(False)
        Mass2.set_visible(False)
        PositionArrow.set_visible(False)
        PositionArrowHead.set_visible(False)
        PositionArrowTail.set_visible(False)
        Force1.set_visible(False)
        Force2.set_visible(False)
        Predicted.set_visible(False)
        Actual.set_visible(False)
        Error.set_visible(False)
        Force1Arrow.set_visible(False)
        Force1ArrowHead.set_visible(False)
        Force2Arrow.set_visible(False)
        Force2ArrowHead.set_visible(False)

        return Spring1,Spring1_left,Spring1_right,Spring2,Spring2_left,Spring2_right,CenterMass,Mass1,Mass2,Force1,Force2,Actual,Error,PositionArrow,PositionArrowHead,PositionArrowTail,Force1Arrow,Force1ArrowHead,Force2Arrow,Force2ArrowHead,

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(Time),10), init_func=init,interval=1, blit=False)
    # if save_as_gif:
    # 	ani.save('test.gif', writer='imagemagick', fps=30)
    plt.show()
