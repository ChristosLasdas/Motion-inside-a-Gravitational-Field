import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## equations of motion (polar coordinates): ##

# d^2 (r) / dt^2 - r * (d(theta) /dt)^2 = - (G * M) / (r^2) 
# 2 * (dr / dt) * (d(theta) / dt) + r * (d^2 (theta) / dt^2) = 0


## transformed equations ##

# dr / dt = vr
# d(vr) / dt = r * omega^2 - G * M / r^2
# d(theta) / dt = omega
# d(omega) / dt = - 2 * vr * omega / r

G = 6.67428e-11  # m^3 / Kg / s^2 
M = 5.972e24     # Kg

initial_position = 6_371_000. + 408_000.    # m
initial_angle = 0.                         # rad

initial_radial_velocity = 0.               # m / s
initial_angular_velocity = 1.1288e-3      # rad / s

n = 40001

t0 = 0
tf = 60*60*24*80.

t = np.linspace(t0,tf,n)

k = (tf - t0) / (n-1.)

r = np.zeros([n],float)
vr = np.zeros([n],float)
theta = np.zeros([n],float)
omega = np.zeros([n],float)

r[0] = initial_position
vr[0] = initial_radial_velocity
theta[0] = initial_angle
omega[0] = initial_angular_velocity

# In what follows, with "A" we denote the values related to r and/or theta and with B the ones dr/dt and/or d theta/dt

def fr(vr, t):
    return vr
    
def fvr(r, omega, t, G=G, M=M):
    return r*omega**2-G*M/r**2

def ftheta(omega, t):
    return omega
    
def fomega(r, vr, omega, t):
    return -2*vr*omega/r


for i in range(1,n):

    Y1_r = fr(vr[i-1], t[i-1])
    Y1_vr = fvr(r[i-1], omega[i-1], t[i-1])
    Y1_theta = ftheta(omega[i-1], t[i-1])
    Y1_omega = fomega(r[i-1], vr[i-1], omega[i-1], t[i-1])

    Y2_r = fr(vr[i-1] + k*Y1_vr/2., t[i-1] + k/2.)
    Y2_vr = fvr(r[i-1] + k*Y1_r/2., omega[i-1] + k*Y1_omega/2., t[i-1] + k/2.)
    Y2_theta = ftheta(omega[i-1] + k*Y1_omega/2., t[i-1] + k/2.)
    Y2_omega = fomega(r[i-1] + k*Y1_r/2., vr[i-1] + k*Y1_vr/2., omega[i-1] + k*Y1_omega/2., t[i-1] + k/2.)

    Y3_r = fr(vr[i-1] + k*Y2_vr/2., t[i-1] + k/2.)
    Y3_vr = fvr(r[i-1] + k*Y2_r/2., omega[i-1] + k*Y2_omega/2., t[i-1] + k/2.)
    Y3_theta = ftheta(omega[i-1] + k*Y2_omega/2., t[i-1] + k/2.)
    Y3_omega = fomega(r[i-1] + k*Y2_r/2., vr[i-1] + k*Y2_vr/2., omega[i-1] + k*Y2_omega/2., t[i-1] + k/2.)

    Y4_r = fr(vr[i-1] + k*Y3_vr, t[i-1] + k)
    Y4_vr = fvr(r[i-1] + k*Y3_r, omega[i-1] + k*Y3_omega, t[i-1] + k)
    Y4_theta = ftheta(omega[i-1] + k*Y3_omega, t[i-1] + k)
    Y4_omega = fomega(r[i-1] + k*Y3_r, vr[i-1] + k*Y3_vr, omega[i-1] + k*Y3_omega, t[i-1] + k)

    r[i] = r[i-1] + (k/6)*(Y1_r + 2*Y2_r + 2*Y3_r + Y4_r)
    vr[i] = vr[i-1] + (k/6)*(Y1_vr + 2*Y2_vr + 2*Y3_vr + Y4_vr)
    theta[i] = theta[i-1] + (k/6)*(Y1_theta + 2*Y2_theta + 2*Y3_theta + Y4_theta)
    omega[i] = omega[i-1] + (k/6)*(Y1_omega + 2*Y2_omega + 2*Y3_omega + Y4_omega)

# trajectory in polar coordinates
plt.figure(1)

plt.axes(projection = 'polar')
plt.polar(theta,r)


# Energy over time in polar coordinates
plt.figure(2)
plt.title('Energy over time in Polar Coordinates')

m = 1

Energy = 1/2*m*(vr**2+r**2 * omega**2)-G*M*m/r

t = np.linspace(t0, tf, n)

plt.plot(t, Energy, label = 'Total Energy')   
      
plt.plot(t, 1/2*m*(vr**2+r**2 * omega**2), label = 'Kinetic Energy')
plt.plot(t, -G*M*m/r, label = 'Potential Energy')

plt.xlim(0, tf)
plt.ylim(np.min(-G*M*m/r),np.max(1/2*m*(vr**2+r**2 * omega**2)))

plt.xlabel('t [s]')
plt.ylabel('E [arbitrary units]')
plt.legend()

plt.show()