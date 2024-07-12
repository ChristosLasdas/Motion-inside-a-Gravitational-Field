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

n = 4001

t0 = 0
tf = 60*60*24*2.

k = (tf - t0) / (n-1.)

r = np.zeros([n],float)
vr = np.zeros([n],float)
theta = np.zeros([n],float)
omega = np.zeros([n],float)

r[0] = initial_position
vr[0] = initial_radial_velocity
theta[0] = initial_angle
omega[0] = initial_angular_velocity

for i in range(1,n):
    r[i] = r[i-1] + k * vr[i-1]
    vr[i] = vr[i-1] + k * (r[i-1] * omega[i-1]**2 - (G * M / r[i-1]**2))         
    theta[i] = theta[i-1] + k * omega[i-1]
    omega[i] = omega[i-1] + k * (-2 * vr[i-1] * omega[i-1] / r[i-1])

# trajectory in polar coordinates
plt.figure(1) 

plt.axes(projection = 'polar')
plt.polar(theta,r)


# Energy over time in polar coordinates
plt.figure(2)
plt.title('Energy over Time in Polar Coordinates')

m = 420_000

Energy = 1/2*m*(vr**2+r**2 * omega**2)-G*M*m/r

t = np.linspace(t0, tf, n)
plt.plot(t, 1/2*m*(vr**2+r**2 * omega**2), label = 'Kinetic Energy')      
plt.plot(t, -G*M*m/r, label = 'Potential Energy')

plt.xlim(0, tf)
plt.ylim(np.min(-G*M*m/r),np.max(1/2*m*(vr**2+r**2 * omega**2)))
plt.plot(t, Energy, label = 'Total Energy')
plt.xlabel('t [s]')
plt.ylabel('E [arbitrary units]')
plt.legend()

plt.show()