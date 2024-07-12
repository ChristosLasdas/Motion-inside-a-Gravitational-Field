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
initial_angular_velocity = 1.5288e-3      # rad / s

n = 400001

t0 = 0
tf = 60*60*24*10.

t = np.linspace(t0,tf,n)


k = (tf - t0) / (n-1.)

r_eu = np.zeros([n],float)
vr_eu = np.zeros([n],float)
theta_eu = np.zeros([n],float)
omega_eu = np.zeros([n],float)

r_eu[0] = initial_position
vr_eu[0] = initial_radial_velocity
theta_eu[0] = initial_angle
omega_eu[0] = initial_angular_velocity

for i in range(1,n):
    r_eu[i] = r_eu[i-1] + k * vr_eu[i-1]
    vr_eu[i] = vr_eu[i-1] + k * (r_eu[i-1] * omega_eu[i-1]**2 - (G * M / r_eu[i-1]**2))         
    theta_eu[i] = theta_eu[i-1] + k * omega_eu[i-1]
    omega_eu[i] = omega_eu[i-1] + k * (-2 * vr_eu[i-1] * omega_eu[i-1] / r_eu[i-1])


#####    4th order Runge-Kutta Method    #####


r_rk = np.zeros([n],float)
vr_rk = np.zeros([n],float)
theta_rk = np.zeros([n],float)
omega_rk = np.zeros([n],float)

r_rk[0] = initial_position
vr_rk[0] = initial_radial_velocity
theta_rk[0] = initial_angle
omega_rk[0] = initial_angular_velocity

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

    Y1_r = fr(vr_rk[i-1], t[i-1])
    Y1_vr = fvr(r_rk[i-1], omega_rk[i-1], t[i-1])
    Y1_theta = ftheta(omega_rk[i-1], t[i-1])
    Y1_omega = fomega(r_rk[i-1], vr_rk[i-1], omega_rk[i-1], t[i-1])

    Y2_r = fr(vr_rk[i-1] + k*Y1_vr/2., t[i-1] + k/2.)
    Y2_vr = fvr(r_rk[i-1] + k*Y1_r/2., omega_rk[i-1] + k*Y1_omega/2., t[i-1] + k/2.)
    Y2_theta = ftheta(omega_rk[i-1] + k*Y1_omega/2., t[i-1] + k/2.)
    Y2_omega = fomega(r_rk[i-1] + k*Y1_r/2., vr_rk[i-1] + k*Y1_vr/2., omega_rk[i-1] + k*Y1_omega/2., t[i-1] + k/2.)

    Y3_r = fr(vr_rk[i-1] + k*Y2_vr/2., t[i-1] + k/2.)
    Y3_vr = fvr(r_rk[i-1] + k*Y2_r/2., omega_rk[i-1] + k*Y2_omega/2., t[i-1] + k/2.)
    Y3_theta = ftheta(omega_rk[i-1] + k*Y2_omega/2., t[i-1] + k/2.)
    Y3_omega = fomega(r_rk[i-1] + k*Y2_r/2., vr_rk[i-1] + k*Y2_vr/2., omega_rk[i-1] + k*Y2_omega/2., t[i-1] + k/2.)

    Y4_r = fr(vr_rk[i-1] + k*Y3_vr, t[i-1] + k)
    Y4_vr = fvr(r_rk[i-1] + k*Y3_r, omega_rk[i-1] + k*Y3_omega, t[i-1] + k)
    Y4_theta = ftheta(omega_rk[i-1] + k*Y3_omega, t[i-1] + k)
    Y4_omega = fomega(r_rk[i-1] + k*Y3_r, vr_rk[i-1] + k*Y3_vr, omega_rk[i-1] + k*Y3_omega, t[i-1] + k)

    r_rk[i] = r_rk[i-1] + (k/6)*(Y1_r + 2*Y2_r + 2*Y3_r + Y4_r)
    vr_rk[i] = vr_rk[i-1] + (k/6)*(Y1_vr + 2*Y2_vr + 2*Y3_vr + Y4_vr)
    theta_rk[i] = theta_rk[i-1] + (k/6)*(Y1_theta + 2*Y2_theta + 2*Y3_theta + Y4_theta)
    omega_rk[i] = omega_rk[i-1] + (k/6)*(Y1_omega + 2*Y2_omega + 2*Y3_omega + Y4_omega)

# trajectory in polar coordinates
plt.figure(1)

plt.axes(projection = 'polar')
plt.polar(theta_eu,r_eu, label = 'Euler')
plt.polar(theta_rk,r_rk, label = 'RK4')
plt.legend()


# Energy over time in polar coordinates
plt.figure(2)

plt.title('Energy as a function of time')

m = 420_000


Energy_eu = 1/2*m*(vr_eu**2+r_eu**2 * omega_eu**2)-G*M*m/r_eu
Energy_rk = 1/2*m*(vr_rk**2+r_rk**2 * omega_rk**2)-G*M*m/r_rk

plt.xlim(0, tf)

plt.plot(t, Energy_eu, label = 'Euler')
plt.plot(t, Energy_rk, label = 'RK4')
plt.xlabel('t [s]')
plt.ylabel('E [arbitrary units]')

# # Kinetic Energy over time in polar coordinates
# plt.title('Kinetic Energy')
# plt.plot(t, 1/2*m*(vr_eu**2+r_eu**2 * omega_eu**2), label = 'Euler')
# plt.plot(t, 1/2*m*(vr_rk**2+r_rk**2 * omega_rk**2), label = 'RK4')

# # Potential Energy over time in polar coordinates
# plt.plot(t, -G*M*m/r_eu, label = 'Euler')    
# plt.plot(t, -G*M*m/r_rk, label = 'RK4', color = 'red')

plt.legend()

plt.show()
