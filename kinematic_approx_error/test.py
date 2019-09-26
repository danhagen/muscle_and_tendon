import numpy as np
import matplotlib.pyplot as plt
import matplotlib
fT = np.linspace(0.0001,1,1001)
cT = 27.8
kT = 0.0048
test = (
    np.sqrt(
        0.25*abs(
            (
                (2*np.exp((2*fT)/(cT*kT)))
                / ((-1 + np.exp(fT/(cT*kT)))**2*cT**3*kT)
                - (2*np.exp((3*fT)/(cT*kT)))
                /((-1 + np.exp(fT/(cT*kT)))**3*cT**3*kT)
            )
            / (
                1 + np.exp((2*fT)/(cT*kT))/((-1 + np.exp(fT/(cT*kT)))**2*cT**2)
            )**(3/2)
        )**2
        + abs(
            -(
                0.5*np.exp(fT/(cT*kT))*((2*np.exp((2*fT)/(cT*kT)))
                / ((-1 + np.exp(fT/(cT*kT)))**2*cT**3*kT)
                - (2*np.exp((3*fT)/(cT*kT)))
                / ((-1 + np.exp(fT/(cT*kT)))**3*cT**3*kT))
            )
            / (
                (-1 + np.exp(fT/(cT*kT)))
                * (
                    1 + np.exp((2*fT)/(cT*kT))/((-1 + np.exp(fT/(cT*kT)))**2*cT**2)
                )**(3/2)*cT
            )
            + np.exp(fT/(cT*kT))
            / (
                (-1 + np.exp(fT/(cT*kT)))
                * np.sqrt(
                    1 + np.exp((2*fT)/(cT*kT))/((-1 + np.exp(fT/(cT*kT)))**2*cT**2)
                )*cT**2*kT
            )
            - np.exp((2*fT)/(cT*kT))
            / (
                (-1 + np.exp(fT/(cT*kT)))**2*np.sqrt(1 + np.exp((2*fT)/(cT*kT))/((-1 + np.exp(fT/(cT*kT)))**2*cT**2))*cT**2*kT
            )
        )**2
    )
    / np.sqrt(
        np.exp((2*fT)/(cT*kT))/(cT**2*(np.exp(fT/(cT*kT)) - 1)**2) + 1)
)
CT = (
    (
        (2**(1./3.)*(4*cT**4 + 6*cT**2))
        / (
            3*cT**2
            * (-abs(
                -16*cT**6
                - 9*cT**4
                + 3*np.sqrt(3) * np.sqrt(
                    160*cT**10
                    - 109*cT**8
                    + 256*cT**6
                )
            )**(1./3.))
        )
    )
    + (
        (-abs(
            -16*cT**6
            - 63*cT**4
            + 3*np.sqrt(3) * np.sqrt(
                160*cT**10
                - 109*cT**8
                + 256*cT**6
            )
        )**(1./3.))
        / (3 * 2**(1/3) * cT**2)
    )
    + 1/3
)
