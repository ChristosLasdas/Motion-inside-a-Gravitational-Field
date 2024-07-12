import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## equations of motion (cartesian coordinates): ##

# d^2 (x) / dt^2 = - (G*M)*x / ((x^2+y^2)^(3/2)) 
# d^2 (y) / dt^2 = - (G*M)*y / ((x^2+y^2)^(3/2))


## transformed equations ##

# dx / dt = xv
# d(vx) / dt = - ( G * M * x ) / ((x^2 + y^2)^(3/2))
# dy / dt = y1
# d(vy) / dt = - ( G * M * y ) / ((x^2 + y^2)^(3/2))


G = 6.67428e-11     # m^3 / Kg / s^2 
M = 5.972e24        # Kg

x_initial_position = 0.      # m
y_initial_position = 408_000. + 6_371_000.        # m

x_initial_velocity = 7778.      # m / s
y_initial_velocity = 0.      # m / s

t0 = 0      # s
tf = 60*60*24*10.     # s
n = 40001

t = np.linspace(t0,tf,n)


#####    Euler Forward Method    #####


x_eu = np.zeros([n])
y_eu = np.zeros([n])
vx_eu = np.zeros([n])
vy_eu = np.zeros([n])

vx_eu[0] = x_initial_velocity
vy_eu[0] = y_initial_velocity
x_eu[0] = x_initial_position
y_eu[0] = y_initial_position

k = (tf-t0)/(n-1)


for i in range(1,n):
    vx_eu[i] = vx_eu[i-1] + k * ( (( -G * M * x_eu[i-1] )) / (( x_eu[i-1]**2 + y_eu[i-1]**2 )**(3/2)) )
    x_eu[i] = x_eu[i-1] + k * ( vx_eu[i-1] ) 
    vy_eu[i] = vy_eu[i-1] + k * ( (( -G * M * y_eu[i-1] )) / (( x_eu[i-1]**2 + y_eu[i-1]**2 )**(3/2)) ) 
    y_eu[i] = y_eu[i-1] + k * ( vy_eu[i-1] )  



#####    4th order Runge-Kutta Method    #####


x_rk = np.zeros([n])
y_rk = np.zeros([n])
vx_rk = np.zeros([n])
vy_rk = np.zeros([n])

vx_rk[0] = x_initial_velocity
vy_rk[0] = y_initial_velocity
x_rk[0] = x_initial_position
y_rk[0] = y_initial_position

def fx(vx, t):
    return vx
    
def fvx(x, y, t, G=G, M=M):
    return - ( G * M * x ) / ((x**2 + y**2)**(3/2))

def fy(vy, t):
    return vy
    
def fvy(x, y, t, G=G, M=M):
    return - ( G * M * y ) / ((x**2 + y**2)**(3/2))


for i in range(1,n):
   
   Y1_x = fx(vx_rk[i-1], t[i-1])
   Y1_y = fy(vy_rk[i-1], t[i-1])
   Y1_vx = fvx(x_rk[i-1], y_rk[i-1], t[i-1])
   Y1_vy = fvy(x_rk[i-1], y_rk[i-1], t[i-1])

   Y2_x = fx(vx_rk[i-1] + k*Y1_vx/2., t[i-1] + k/2.)
   Y2_y = fy(vy_rk[i-1] + k*Y1_vy/2., t[i-1] + k/2.)
   Y2_vx = fvx(x_rk[i-1] + k*Y1_x/2., y_rk[i-1] + k*Y1_y/2., t[i-1] + k/2.)
   Y2_vy = fvy(x_rk[i-1] + k*Y1_x/2., y_rk[i-1] + k*Y1_y/2., t[i-1] + k/2.)

   Y3_x = fx(vx_rk[i-1] + k*Y2_vx/2., t[i-1] + k/2.)
   Y3_y = fy(vy_rk[i-1] + k*Y2_vy/2., t[i-1] + k/2.)
   Y3_vx = fvx(x_rk[i-1] + k*Y2_x/2., y_rk[i-1] + k*Y2_y/2., t[i-1] + k/2.)
   Y3_vy = fvy(x_rk[i-1] + k*Y2_x/2., y_rk[i-1] + k*Y2_y/2., t[i-1] + k/2.)
   
   Y4_x = fx(vx_rk[i-1] + k*Y3_vx, t[i-1])
   Y4_y = fy(vy_rk[i-1] + k*Y3_vy, t[i-1])
   Y4_vx = fvx(x_rk[i-1] + k*Y3_x, y_rk[i-1]+k*Y3_y, t[i-1])
   Y4_vy = fvy(x_rk[i-1] + k*Y3_x, y_rk[i-1] + k*Y3_y, t[i-1])
   
   x_rk[i] = x_rk[i-1] + (k/6)*(Y1_x + 2*Y2_x + 2*Y3_x + Y4_x)
   y_rk[i] = y_rk[i-1]+(k/6)*(Y1_y+2*Y2_y+2*Y3_y+Y4_y)
   vx_rk[i] = vx_rk[i-1] + (k/6)*(Y1_vx + 2*Y2_vx + 2*Y3_vx + Y4_vx)
   vy_rk[i] = vy_rk[i-1] + (k/6)*(Y1_vy + 2*Y2_vy + 2*Y3_vy + Y4_vy)


# trajectory comparison in cartesian coordinates
plt.figure(1)
plt.title('Trajectory comparison in Cartesian Coordinates')

plt.axes().set_aspect('equal', 'datalim')  # scaling constrained

plt.plot(x_eu, y_eu, linewidth = 1.2, label = 'Euler')
plt.plot(x_rk, y_rk, linewidth = 1.2, label = 'RK4', color = 'red')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim(-np.max(np.abs(x_eu)), np.max(np.abs(x_eu)))
plt.ylim(-np.max(np.abs(y_eu)), np.max(np.abs(y_eu)))
plt.legend()
 
plt.axhline(y=0, color = 'black', linestyle = '-')  # plot the coordinate axes
plt.axvline(x=0, color = 'black', linestyle = '-')


# position comparison as a function of time
plt.figure(2)  
plt.title('Positions as time progresses')

plt.plot(t, x_eu, '.', label = 'x-euler')
#plt.plot(t, y_eu, '.', label = 'y_euler')
plt.plot(t, x_rk, '.', label = 'x-RK4', color = 'red')
#plt.plot(t, y_rk, '.', label = 'y_RK4', color = 'red')

plt.legend(loc="lower left")
plt.xlabel('t [s]')
plt.ylabel('position [m]')



# velocity comparison as a function of time
plt.figure(3)
plt.title('Velocity comparison')

# plt.plot(t,vx_eu, '.', label = 'x-velocity-euler')
# plt.plot(t,vx_rk, '.', label = 'x-velocity-RK4')

plt.plot(t,vy_eu, '.', label = 'y-velocity-euler')
plt.plot(t,vy_rk, '.', label = 'y-velocity-RK4', color = 'red')

plt.xlabel('t [s]')
plt.ylabel('v [m/s]')

plt.legend(loc = 'lower right')

# Energy as a function of time 
plt.figure(4)
plt.title('Energy as a function of time')

m = 420_000     # Kg

Energy_eu = 1/2*m*(vx_eu**2+vy_eu**2)-G*M*m/np.sqrt(x_eu**2+y_eu**2)
#plt.plot(t, 1/2*m*(vx_eu**2+vy_eu**2), label = 'Kinetic Energy (Euler)')   # Kinetic Energy (Euler Method) 
#plt.plot(t, -G*M*m/np.sqrt(x_eu**2+y_eu**2), label = 'Potential Energy (Euler)')   # Potential Energy (Euler Method)

Energy_rk = 1/2*m*(vx_rk**2+vy_rk**2)-G*M*m/np.sqrt(x_rk**2+y_rk**2)
#plt.plot(t, 1/2*m*(vx_eu**2+vy_eu**2), label = 'Kinetic Energy (RK4)')   # Kinetic Energy (RK4 Method)
#plt.plot(t, -G*M*m/np.sqrt(x_eu**2+y_eu**2), label = 'Potential Energy (RK4)')   # Potential Energy (RK4 Method)

plt.plot(t, Energy_eu, label = 'Euler')
plt.plot(t, Energy_rk, label = 'RK4', color = 'red')
plt.xlabel('t [s]')
plt.ylabel('E [arbirary units]')

plt.legend()

plt.show()