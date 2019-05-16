import numpy as np

Amp = 7.5*np.pi/180
Base = 90*np.pi/180
Freq = 2*np.pi

# N_seconds = 1
# N = N_seconds*10000 + 1
# t = np.linspace(0,N_seconds,N)
# dt = t[1]-t[0]

### Reference Trajectory ###

r = lambda t: Amp*np.cos(Freq*t) + Base
dr = lambda t: -Amp*Freq*np.sin(Freq*t)
d2r = lambda t: -Amp*Freq**2*np.cos(Freq*t)
d3r = lambda t: Amp*Freq**3*np.sin(Freq*t)
d4r = lambda t: Amp*Freq**4*np.cos(Freq*t)

############################
