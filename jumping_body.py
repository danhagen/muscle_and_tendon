import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle,Circle

density = 100 # kg/m²

# Body params
l_b = 0.2
w_b = l_b/4
m_b = (np.pi*w_b**2/4 + l_b*w_b)*density

# Proximal Limb params
l_p = 0.1
w_p = l_p/4
m_p = (np.pi*w_p**2/4 + l_p*w_p)*density

# Intermediate Limb params
l_i = 0.1
w_i = l_i/4
m_i = (np.pi*w_i**2/4 + l_i*w_i)*density

# Distal Limb params
l_d = 0.05
w_d = l_d/4
m_d = (np.pi*w_d**2/4 + l_d*w_d)*density

r_b = lambda x,y: np.matrix([[x],[y]])
r_p = lambda x,y,theta_b,theta_p: (
    r_b(x,y)
    - np.matrix([
        [l_b*np.sin(theta_b)/2],
        [l_b*np.cos(theta_b)/2]
    ])
    - np.matrix([
        [l_p*np.sin(theta_b-theta_p)/2],
        [l_p*np.cos(theta_b-theta_p)/2]
    ])
)
r_i = lambda x,y,theta_b,theta_p,theta_i: (
    r_p(x,y,theta_b,theta_p)
    - np.matrix([
        [l_p*np.sin(theta_b-theta_p)/2],
        [l_p*np.cos(theta_b-theta_p)/2]
    ])
    - np.matrix([
        [l_i*np.sin(theta_b-theta_p+theta_i)/2],
        [l_i*np.cos(theta_b-theta_p+theta_i)/2]
    ])
)
r_d = lambda x,y,theta_b,theta_p,theta_i,theta_d: (
    r_i(x,y,theta_b,theta_p,theta_i)
    - np.matrix([
        [l_i*np.sin(theta_b-theta_p+theta_i)/2],
        [l_i*np.cos(theta_b-theta_p+theta_i)/2]
    ])
    - np.matrix([
        [l_d*np.sin(theta_b-theta_p+theta_i-theta_d)/2],
        [l_d*np.cos(theta_b-theta_p+theta_i-theta_d)/2]
    ])
)

r_c = lambda x,y,theta_b,theta_p,theta_i,theta_d: (
    (
        m_b*r_b(x,y)
        + m_p*r_p(x,y,theta_b,theta_p)
        + m_i*r_i(x,y,theta_b,theta_p,theta_i)
        + m_d*r_d(x,y,theta_b,theta_p,theta_i,theta_d)
    )
    / (m_b + m_p + m_i + m_d)
)

x,y = 0,0.45
theta_b = 30*np.pi/180
theta_p = 30*np.pi/180
theta_i = 30*np.pi/180
theta_d = 30*np.pi/180

plt.figure()
ax = plt.gca()
ax.set_aspect('equal')

x_b,y_b = r_b(x,y)
x_b,y_b = x_b[0,0],y_b[0,0]

x_p,y_p = r_p(x,y,theta_b,theta_p)
x_p,y_p = x_p[0,0],y_p[0,0]

x_i,y_i = r_i(x,y,theta_b,theta_p,theta_i)
x_i,y_i = x_i[0,0],y_i[0,0]

x_d,y_d = r_d(x,y,theta_b,theta_p,theta_i,theta_d)
x_d,y_d = x_d[0,0],y_d[0,0]

x_c,y_c = r_c(x,y,theta_b,theta_p,theta_i,theta_d)
x_c,y_c = x_c[0,0],y_c[0,0]

Body_top = Circle(
    (
        x_b+l_b*np.sin(theta_b)/2,
        y_b+l_b*np.cos(theta_b)/2
    ),
    w_b/2,
    fill=True,
    facecolor='#B0D0DF'
)
ax.add_patch(Body_top)

Body_bottom = Circle(
    (
        x_b-l_b*np.sin(theta_b)/2,
        y_b-l_b*np.cos(theta_b)/2
    ),
    w_b/2,
    fill=True,
    facecolor='#B0D0DF'
)
ax.add_patch(Body_bottom)

Body = Rectangle(
    (
        x_b-l_b*np.sin(theta_b)/2-w_b*np.cos(theta_b)/2,
        y_b-l_b*np.cos(theta_b)/2+w_b*np.sin(theta_b)/2
    ),
    w_b,
    l_b,
    angle=-theta_b*180/np.pi,
    fill=True,
    facecolor='#B0D0DF'
)
ax.add_patch(Body)

ProximalLimbJoint = Circle(
    (
        x_p+l_p*np.sin(theta_b-theta_p)/2,
        y_p+l_p*np.cos(theta_b-theta_p)/2
    ),
    w_p/2,
    fill=True,
    color='#96B2C0'
)
ax.add_patch(ProximalLimbJoint)

ProximalLimb = Rectangle(
    (
        x_p-l_p*np.sin(theta_b-theta_p)/2-w_p*np.cos(theta_b-theta_p)/2,
        y_p-l_p*np.cos(theta_b-theta_p)/2+w_p*np.sin(theta_b-theta_p)/2
    ),
    w_p,
    l_p,
    angle=-(theta_b-theta_p)*180/np.pi,
    fill=True,
    color='#96B2C0'
)
ax.add_patch(ProximalLimb)


IntermediateLimbJoint = Circle(
    (
        x_p-l_p*np.sin(theta_b-theta_p)/2,
        y_p-l_p*np.cos(theta_b-theta_p)/2
    ),
    w_i/2,
    fill=True,
    color='#B0D0DF'
)
ax.add_patch(IntermediateLimbJoint)

IntermediateLimb = Rectangle(
    (
        x_i-l_i*np.sin(theta_b-theta_p+theta_i)/2-w_i*np.cos(theta_b-theta_p+theta_i)/2,
        y_i-l_i*np.cos(theta_b-theta_p+theta_i)/2+w_i*np.sin(theta_b-theta_p+theta_i)/2
    ),
    w_i,
    l_i,
    angle=-(theta_b-theta_p+theta_i)*180/np.pi,
    fill=True,
    color='#B0D0DF'
)
ax.add_patch(IntermediateLimb)
IntermediateLimbJoint2 = Circle(
    (
        x_i-l_i*np.sin(theta_b-theta_p+theta_i)/2,
        y_i-l_i*np.cos(theta_b-theta_p+theta_i)/2
    ),
    w_i/2,
    fill=True,
    color='#B0D0DF'
)
ax.add_patch(IntermediateLimbJoint2)

DistalLimbJoint = Circle(
    (
        x_i-l_i*np.sin(theta_b-theta_p+theta_i)/2,
        y_i-l_i*np.cos(theta_b-theta_p+theta_i)/2
    ),
    w_d/2,
    fill=True,
    color='#96B2C0'
)
ax.add_patch(DistalLimbJoint)

DistalLimb = Rectangle(
    (
        x_d-l_d*np.sin(theta_b-theta_p+theta_i-theta_d)/2-w_d*np.cos(theta_b-theta_p+theta_i-theta_d)/2,
        y_d-l_d*np.cos(theta_b-theta_p+theta_i-theta_d)/2+w_d*np.sin(theta_b-theta_p+theta_i-theta_d)/2
    ),
    w_d,
    l_d,
    angle=-(theta_b-theta_p+theta_i-theta_d)*180/np.pi,
    fill=True,
    color='#96B2C0'
)
ax.add_patch(DistalLimb)

DistalLimbJoint2 = Circle(
    (
        x_d-l_d*np.sin(theta_b-theta_p+theta_i-theta_d)/2,
        y_d-l_d*np.cos(theta_b-theta_p+theta_i-theta_d)/2
    ),
    w_d/2,
    fill=True,
    color='#96B2C0'
)
ax.add_patch(DistalLimbJoint2)

COM = Circle(
    (
        x_c,
        y_c
    ),
    w_d/2,
    fill=True,
    color='g'
)
ax.add_patch(COM)

ax.set_xlim([-(l_b+l_p+l_i+l_d),(l_b+l_p+l_i+l_d)])
ax.set_ylim([-(l_b+l_p+l_i+l_d),(l_b+l_p+l_i+l_d)])
