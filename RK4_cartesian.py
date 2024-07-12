import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## equations of motion (cartesian coordinates): ##

# d^2 (x) / dt^2 = - (G*M)*x / ((x^2+y^2)^(3/2)) 
# d^2 (y) / dt^2 = - (G*M)*y / ((x^2+y^2)^(3/2))


## transformed equations  ##

# dx / dt = x1
# dx1 / dt = - ( G * M * x ) / ((x^2 + y^2)^(3/2))
# dy / dt = y1
# dy1 / dt = - ( G * M * y ) / ((x^2 + y^2)^(3/2))

G = 6.67428e-11     # m^3 / Kg / s^2 
M = 5.972e24        # Kg

x_initial_position = 0.      # m
y_initial_position = 6_371_000. + 408_000.        # m

x_initial_velocity = 7778.      # m / s
y_initial_velocity = 0.     # m / s

t0 = 0      # s
tf = 60*60*24*5.     # s
n = 4001

t = np.linspace(t0,tf,n)

x = np.zeros([n])
y = np.zeros([n])
vy = np.zeros([n])
vx = np.zeros([n])

vx[0] = x_initial_velocity
vy[0] = y_initial_velocity
x[0] = x_initial_position
y[0] = y_initial_position

k = (tf-t0)/(n-1)


def fx(vx, t):
    return vx
    
def fvx(x, y, t, G=G, M=M):
    return - ( G * M * x ) / ((x**2 + y**2)**(3/2))

def fy(vy, t):
    return vy
    
def fvy(x, y, t, G=G, M=M):
    return - ( G * M * y ) / ((x**2 + y**2)**(3/2))


for i in range(1,n):
   
   Y1_x = fx(vx[i-1], t[i-1])
   Y1_y = fy(vy[i-1], t[i-1])
   Y1_vx = fvx(x[i-1], y[i-1], t[i-1])
   Y1_vy = fvy(x[i-1], y[i-1], t[i-1])

   Y2_x = fx(vx[i-1] + k*Y1_vx/2., t[i-1] + k/2.)
   Y2_y = fy(vy[i-1] + k*Y1_vy/2., t[i-1] + k/2.)
   Y2_vx = fvx(x[i-1] + k*Y1_x/2., y[i-1] + k*Y1_y/2., t[i-1] + k/2.)
   Y2_vy = fvy(x[i-1] + k*Y1_x/2., y[i-1] + k*Y1_y/2., t[i-1] + k/2.)

   Y3_x = fx(vx[i-1] + k*Y2_vx/2., t[i-1] + k/2.)
   Y3_y = fy(vy[i-1] + k*Y2_vy/2., t[i-1] + k/2.)
   Y3_vx = fvx(x[i-1] + k*Y2_x/2., y[i-1] + k*Y2_y/2., t[i-1] + k/2.)
   Y3_vy = fvy(x[i-1] + k*Y2_x/2., y[i-1] + k*Y2_y/2., t[i-1] + k/2.)
   
   Y4_x = fx(vx[i-1] + k*Y3_vx, t[i-1])
   Y4_y = fy(vy[i-1] + k*Y3_vy, t[i-1])
   Y4_vx = fvx(x[i-1] + k*Y3_x, y[i-1]+k*Y3_y, t[i-1])
   Y4_vy = fvy(x[i-1] + k*Y3_x, y[i-1] + k*Y3_y, t[i-1])
   
   x[i] = x[i-1] + (k/6)*(Y1_x + 2*Y2_x + 2*Y3_x + Y4_x)
   y[i] = y[i-1]+(k/6)*(Y1_y+2*Y2_y+2*Y3_y+Y4_y)
   vx[i] = vx[i-1] + (k/6)*(Y1_vx + 2*Y2_vx + 2*Y3_vx + Y4_vx)
   vy[i] = vy[i-1] + (k/6)*(Y1_vy + 2*Y2_vy + 2*Y3_vy + Y4_vy)
  

   
# trajectory in cartesian coordinates
plt.figure(1)
plt.title('Trajectory in Cartesian Coordinates')

plt.axes().set_aspect('equal', 'datalim')  # scaling constrained
plt.plot(x, y, linewidth = 1.2)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim(-np.max(np.abs(x)), np.max(np.abs(x)))
plt.ylim(-np.max(np.abs(y)), np.max(np.abs(y)))

# plot the coordinate axes 
plt.axhline(y=0, color = 'black', linestyle = '-')
plt.axvline(x=0, color = 'black', linestyle = '-')
plt.title('Trajectory in Cartesian Coordinates')


# position as functions of time
plt.figure(2)  
plt.title('Positions as functions of Time')

plt.plot(t, x, '.', color='black', label = 'x-position')
plt.plot(t, y, '.', color='green', label = 'y-position')
plt.legend(loc="lower left")
plt.xlabel('t [s]')
plt.ylabel('Position [m]')

# velocity as functions of time
plt.figure(3)
plt.title('Velocities as functions of Time')

plt.plot(t,vx, '.', color='red', label = 'x-velocity')
plt.plot(t,vy, '.', color='blue', label = 'y-velocity')
plt.legend(loc="lower right")
plt.xlabel('t [s]')
plt.ylabel('Velocity [m/s]')

# Energy as a function of time 
plt.figure(4)
plt.title('Energy as a function of time')

m = 420_000     # Kg

Energy = 1/2*m*(vx**2+vy**2)-G*M*m/np.sqrt(x**2+y**2)

plt.plot(t, Energy, label = 'Total Energy')
plt.plot(t, 1/2*m*(vx**2+vy**2), label = 'Kinetic Energy')
plt.plot(t, -G*M*m/np.sqrt(x**2+y**2), label = 'Potential Energy')

plt.xlabel('t [s]')
plt.ylabel('E [arbitrary units]')
plt.legend()

plt.show()

# animation of the trajectory
fig, ax = plt.subplots()

trajectory = ax.plot(x[0], y[0])[0]   # first frame of the animated plot

ax.set_aspect('equal', 'datalim')   # set the aspect of x and y axes to be 1
ax.set_ylim(-np.max(np.abs(y)),np.max(np.abs(y)))
ax.set_title('Animated Trajectory in Cartesian Coordinates')

# plot the coordinate axes 
ax.axhline(y=0, color = 'black', linestyle = '-')
ax.axvline(x=0, color = 'black', linestyle = '-')

# define a function that is to be updated in each frame
def update(frame):
    # update the trajectory plot:
    trajectory.set_xdata(x[:frame])
    trajectory.set_ydata(y[:frame])
    return (trajectory)

ani = animation.FuncAnimation(fig=fig, func=update, frames=n, interval=0.1)
plt.show()