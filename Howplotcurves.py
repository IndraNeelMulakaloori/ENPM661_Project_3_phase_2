import matplotlib.pyplot as plt
import numpy as np
import math

## for turtle3 waffle pi
# Wheel Radius (R): 33 mm
# Robot Radius (r): 220 mm
# Wheel Distance (L): 287 mm

fig, ax = plt.subplots()

def plot_curve(Xi,Yi,Thetai,UL,UR):
    t = 0
    r = 0.038
    L = 0.354
    dt = 0.1
    Xn=Xi
    Yn=Yi
    Thetan = 3.14 * Thetai / 180


# Xi, Yi,Thetai: Input point's coordinates
# Xs, Ys: Start point coordinates for plot function
# Xn, Yn, Thetan: End point coordintes
    D=0
    while t<1:
        t = t + dt
        Xs = Xn
        Ys = Yn
        Xn += 0.5*r * (UL + UR) * math.cos(Thetan) * dt
        Yn += 0.5*r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        plt.plot([Xs, Xn], [Ys, Yn], color="blue")
        
    Thetan = 180 * (Thetan) / 3.14
    return Xn, Yn, Thetan, D
    

actions=[[5,5], [10,10],[5,0],[0,5],[5,10],[10,5]]
        
for action in actions:
   X1= plot_curve(0,0,45, action[0],action[1]) # (0,0,45) hypothetical start configuration
   for action in actions:
      X2=plot_curve(X1[0],X1[1],X1[2], action[0],action[1])
      
   

  

plt.grid()

ax.set_aspect('equal')

plt.xlim(0,1)
plt.ylim(0,1)

plt.title('How to plot a vector in matplotlib ?',fontsize=10)

plt.show()
plt.close()
    
