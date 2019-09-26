import numpy as np
import matplotlib.pyplot as plt

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

kT_low = 0.001
kT_high = kT_intersection

cT_low = 40
cT_high = 100

lT_array = np.linspace(0,1.1/l_Ts_over_l_To,1001)

fT_low = cT_high*kT_low*np.log(
    np.exp(
        (l_Ts_over_l_To*lT_array - 1 + 1/cT_high)
        /kT_low
    )
    + 1
)
lT_low_initial = (1/l_Ts_over_l_To)*(
    kT_low*np.log(
        np.exp(initial_force/(cT_high*kT_low))
        - 1
    )
    + (1-1/cT_high)
)

fT_high = cT_low*kT_high*np.log(
    np.exp(
        (l_Ts_over_l_To*lT_array - 1 + 1/cT_low)
        /kT_high
    )
    + 1
)
lT_high_initial = (1/l_Ts_over_l_To)*(
    kT_high*np.log(
        np.exp(initial_force/(cT_low*kT_high))
        - 1
    )
    + (1-1/cT_low)
)

plt.figure(figsize=(6,8))
ax=plt.gca()
ax.set_xlabel(r"$\hat{l}_T$")
ax.set_xlim([1,1/l_Ts_over_l_To+0.02])
ax.set_xticks([1,lT_high_initial,lT_low_initial,1/l_Ts_over_l_To])
ax.set_xticklabels(["1",r"$\hat{l}_{T}(t_o)$",r"$\hat{l}_{T}(t_o)$",r"$l_{T,o}/l_{T,s}$"])

ax.set_ylim([0,1.15])
ax.set_ylabel(r"$\hat{f}_T$")
ax.set_yticks([0,initial_force,1])
ax.set_yticklabels(["",r"$\hat{f}_T(t_o)$","1"])

ax.plot(lT_array,fT_low,'b',lw=2)
ax.plot(lT_array,fT_high,'r',lw=2)
ax.plot([1,1/l_Ts_over_l_To,1/l_Ts_over_l_To],[1,1,0],'k--')
ax.plot([1,1/l_Ts_over_l_To],[initial_force,initial_force],'k--')
ax.plot([lT_high_initial,lT_high_initial],[0,initial_force],'r--')
ax.plot([lT_low_initial,lT_low_initial],[0,initial_force],'b--')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position('zero')

ax.get_xticklabels()[1].set_color('red')
ax.get_xticklabels()[2].set_color('blue')
plt.show()
