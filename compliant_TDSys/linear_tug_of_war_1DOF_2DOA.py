import sympy as sp
import numpy as np
from control import *
import matplotlib.pyplot as plt

# Motor parameters
r_m = 0.01 # m
m_m = 0.1 # kg
I_m = 0.5*m_m*r_m**2 # kg⋅m² = N⋅m⋅s²
k_m = 0.01 # N/m
b_m = 0.001 # N⋅s/m
MaxTorq = 10 # N

#>>>>>>> Pulley 1 parameters <<<<<<<#
r_p1 = 0.05 # m
m_p1 = 0.1 # kg
I_p1 = 0.5*m_p1*r_p1**2 # kg⋅m² = N⋅m⋅s²

#>>>>>>> Pulley 2 parameters <<<<<<<#
r_p2 = 0.05 # m
m_p2 = 0.1 # kg
I_p2 = 0.5*m_p2*r_p2**2 # kg⋅m² = N⋅m⋅s²

#>>>>>>> Center mass parameters <<<<<<<#
M = 1 # kg
k_M = 0.01 # N/m
b_M = 0.001 # N⋅s/m

#>>>>>>> Tendon stiffness parameters <<<<<<<#
E_tendon = 1e7 # 0.01 - 0.1 GPa = 1e7-1e8 N/m²
d_tendon = 0.002 # m
A_tendon = np.pi*(d_tendon/2)**2 # m²

#>>>>>>> Tendon 1 stiffness (as a function of slack length) <<<<<<<#
l_1s = 0.05 # m (tendon 1 slack length)
k_1 = (E_tendon*A_tendon/l_1s) # N/m

#>>>>>>> Tendon 2 stiffness (as a function of slack length) <<<<<<<#
l_2s = 0.05 # m (tendon 2 slack length)
k_2 = (E_tendon*A_tendon/l_2s) # N/m

#>>>>>>> Torsion between motor and pulley (negligible, hopefully!) <<<<<<<#
r_r = 0.005 # m
J_r = 0.5*np.pi*r_r**4 # m⁴ (Polar Moment of Inertia for a cylinder of radius r_r)
G = 5e10 # Steel : 79.3 GPa = 7.93e10 N/m²
l_r = 0.01 # m
k_t = G*J_r/l_r

#>>>>>>> Target <<<<<<<#
X_p = np.array([0,0,0,0,0,0,0,0,0,0])

#>>>>>>> State Space Matrices <<<<<<<#
A = np.matrix(
    [
        [0,1,0,0,0,0,0,0,0,0],
        [-(k_M+k_1+k_2)/M,-b_M/M,-k_1*r_p1/M,0,k_2*r_p2/M,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [-r_p1*k_1/I_p1,0,-(r_p1**2*k_1+k_t)/I_p1,0,0,0,k_t/I_p1,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [r_p2*k_2/I_p2,0,0,0,-(r_p2**2*k_2+k_t)/I_p2,0,0,0,k_t/I_p2,0],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,k_t/I_m,0,0,0,-(k_m+k_t)/I_m,-b_m/I_m,0,0],
        [0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,k_t/I_m,0,0,0,-(k_m+k_t)/I_m,-b_m/I_m]
    ]
)
# NOTE: We have normalized the input between 0 and 1 to simply check if the movement is within the limits of the motors maximum torque output.

B = np.matrix(
    [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [MaxTorq/I_m,0],
        [0,0],
        [0,MaxTorq/I_m]
    ]
)

C = np.matrix([[1,0,0,0,0,0,0,0,0,0]])
D = np.matrix([[0,0]])

#>>>>>>> Test for Controllability <<<<<<<#
evalues = np.linalg.eig(A)[0]
assert (
        np.array(
            [
                np.linalg.matrix_rank(
                    np.concatenate([evalues[i]*np.eye(10)-A,B],axis=1)
                )
                for i in range(len(evalues))
            ]
        )
        ==10).all(), \
    "PBH test failed. Check E-V to see which mode is not controllable."

#>>>>>>> Test for Stabilizability <<<<<<<#
unstable_evalues = evalues[np.where(np.real(evalues)>0)]
if len(unstable_evalues)>0:
    assert (
        np.array(
            [
                np.linalg.matrix_rank(
                    np.concatenate([unstable_evalues[i]*np.eye(10)-A,B],axis=1)
                )
                for i in range(len(unstable_evalues))
            ]
        )
        ==10).all(), \
    "PBH test failed. Check E-V to see which mode is not stabilizable."

#>>>>>>> Test for Detectability <<<<<<<#
if len(unstable_evalues)>0:
    assert (
        np.array(
            [
                np.linalg.matrix_rank(
                    np.concatenate([unstable_evalues[i]*np.eye(10)-A.T,C.T],axis=1)
                )
                for i in range(len(unstable_evalues))
            ]
        )
        ==10).all(), \
    "PBH test failed. Check E-V to see which mode is not detectable."

#>>>>>>> Test for Observability <<<<<<<#
# evalues = np.linalg.eig(A.T)[0]
# assert (
#         np.array(
#             [
#                 np.linalg.matrix_rank(
#                     np.concatenate([evalues[i]*np.eye(10)-A.T,C.T],axis=1)
#                 )
#                 for i in range(len(evalues))
#             ]
#         )
#         ==10).all(), \
#     "PBH test failed. Check E-V to see which mode is not observable."
#>>>>>>> SYSTEM IS NOT OBSERVABLE FOR LAST 4 E-V. <<<<<<<#

Q = np.diag([1]*10)
R = np.diag([1,1])

(K,X,E) = lqr(A,B,Q,R)

Time = np.linspace(0,1,1001)
dt = Time[1]-Time[0]
X_o = [0.1,0,0,0,0,0,0,0,0,0]
X = np.zeros((10,len(Time)))
X_fb = np.zeros((10,len(Time)))
X[:,0] = X_o
X_fb[:,0] = X_o
for i in range(len(Time)-1):
    # X[:,i+1] = np.dot(np.eye(10,10)+A*dt,X[:,i])
    X[:,i+1] = X[:,i] + np.array(A*np.matrix(X[:,i,np.newaxis])).T*dt
    # X_fb[:,i+1] = np.dot(np.eye(10,10)+(A+B*K)*dt,X_fb[:,i])
    X_fb[:,i+1] = X_fb[:,i] + np.array((A-B*K)*np.matrix(X_fb[:,i,np.newaxis])).T*dt


fig1 = plt.figure()
ax1 = plt.gca()
ax1.set_title("Unforced Response")
ax1.plot(Time,X[0,:],'r')
ax1.plot([Time[0],Time[-1]],[X_p[0],X_p[0]],'k--')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Position (m)")
ax1.set_ylim(-2,2)

fig2 = plt.figure()
ax2 = plt.gca()
ax2.set_title("Feedback Control")
ax2.plot(Time,X_fb[0,:],'r')
ax2.plot([Time[0],Time[-1]],[X_p[0],X_p[0]],'k--')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (m)")
ax2.set_ylim(-2,2)

plt.show()
