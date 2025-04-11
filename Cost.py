import matplotlib.pyplot as plt
import numpy as np
import math

## turtle 3 wafflepi 
# Wheel Radius (R): 33 mm
# Robot Radius (r): 220 mm
# Wheel Distance (L): 287 mm

def cost(Xi,Yi,Thetai,UL,UR):
   t = 0
   r = 0.038
   L = 0.354
   dt = 0.1
   Xn=Xi
   Yn=Yi
   Thetan = 3.14 * Thetai / 180
   # Xi, Yi,Thetai: Input point's coordinates
   # Xs, Ys: Start point coordinates for plot function
   # Xn, Yn, Thetan: End point coordinates
   D=0
   while t<1:
        t = t + dt
        Delta_Xn = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
        Delta_Yn = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        Xn += Delta_Xn
        Yn += Delta_Yn
        D=D+ math.sqrt(math.pow((0.5*r * (UL + UR) * math.cos(Thetan) * dt),2)+math.pow((0.5*r * (UL + UR) * math.sin(Thetan) * dt),2))
   Thetan = 180 * (Thetan) / 3.14
   return Xn, Yn, Thetan, D
    
def cost_numpy(Xi,Yi,Thetai, UL, UR):
    r = 0.038  # Wheel radius
    L = 0.354  # Distance between wheels

    # Time array
    t = np.linspace(0, 1, 11)  # 11 steps = 10 segments
    dt = t[1] - t[0]

    # Angular velocity
    w = (r / L) * (UR - UL)
    Thetan = np.deg2rad(Thetai) + w * t

    # Linear velocity
    v = 0.5 * r * (UL + UR)

    # Compute deltas
    dx = v * np.cos(Thetan) * dt
    dy = v * np.sin(Thetan) * dt

    # Integrate position
    x_points = np.cumsum(dx) + Xi
    y_points = np.cumsum(dy) + Yi

    # Total distance traveled
    total_distance = np.sum(np.hypot(np.diff(x_points), np.diff(y_points)))

    # Final orientation
    final_theta = int(np.rad2deg(Thetan[-1])) % 360

    # Return final state + trajectory (optional)
    return x_points[-1], y_points[-1], final_theta, total_distance, list(zip(x_points, y_points))

actions=[[5,5], [10,10],[5,0],[0,5],[5,10],[10,5]]
        
for action in actions:
     k=cost(0,0,45, action[0],action[1]) 
     k_numpy=cost_numpy(0,0,45, action[0],action[1])
     print(f"Action: {action}, Cost: {k}, Cost Numpy: {k_numpy}")
  
 

    
