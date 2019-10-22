import numpy as np
import matplotlib.pyplot as plt

params = {
    "kT" : 0.003,
    "cT" : 50,
    "pa" : 0,
    "l_Ts_over_l_mo" : 1,
    "l_Ts_over_l_To" : 0.95
}
def return_error(fT,fTo,**params):
    kT = params.get("kT",0.003)
    cT = params.get("cT",50)
    pa = params.get("pa",0)
    l_Ts_over_l_mo = params.get("l_Ts_over_l_mo",1)
    l_Ts_over_l_To = params.get("l_Ts_over_l_To",0.95)
    error =  (
        - l_Ts_over_l_mo/l_Ts_over_l_To
        * kT / np.cos(pa)
        * np.log(
            (
                np.exp(fT/(cT*kT)) - 1
            )
            / (
                np.exp(fTo/(cT*kT)) - 1
            )

        )
    )
    lTo = (
        kT * np.log(
            np.exp(fTo/(cT*kT))
            - 1
        )
        + (1 - 1/cT)
    ) / l_Ts_over_l_To
    return(error,lTo)

fT = np.linspace(1e-8,1.25,1001)
fTo1 = 0.1
fTo2 = 0.5

error1,lTo1 = return_error(fT,fTo1,**params)
error2,lTo2 = return_error(fT,fTo2,**params)

b1 = (
    params["l_Ts_over_l_mo"]
    * (lTo1-1/params["l_Ts_over_l_To"])
    / np.cos(params["pa"])
)
b2 = (
    params["l_Ts_over_l_mo"]
    * (lTo2-1/params["l_Ts_over_l_To"])
    / np.cos(params["pa"])
)

plt.figure()
ax = plt.gca()
plt.plot(fT,error1,'b')
plt.plot(fT,error2,'r')
plt.plot(
    [0,1,1],
    [b1,b1,0],
    'k--'
)
plt.plot(
    [0,1],
    [b2,b2],
    'k--'
)
ax.set_xlabel(r"$\hat{f}_T$")
ax.set_ylabel(r"$\eta$")
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([1.25*min(b1,b2),-1.25*min(b1,b2)])
ax.set_xlim([-0.25,1.25])

plt.show()
