import pandas as pd
import cvxpy as cp
import mosek
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import random
from scipy.integrate import cumtrapz


# Constants ==================================================
# powertrain constant to define

## Shared across platforms
m = 1611+82             # average vehicle mass + driver [kg]
rho = 1.225             # air density [kg/m^3]
Cd = 0.23               # drag coefficient
Aref = 2.22             # average vehicle frontal area [m^2]
g = 9.81                # gravity constant [m/s^2]
Crr = 0.01              # rolling resistance coefficient
Cv = 0.3                # kinetic friction coefficient
r = 0.334               # wheel radius [m] (235/45R18 tire)

## ICE only (Honda Civic 2012 5AT)
f_ice = 0.375           # idle fuel consumption [ml/s]
P_max = 104.398         # max engine power [kw]
P_min = 0               # min engine power [kw]
wp_max = 6500           # engine angular speed at max power [rpm]
wt_max = 4300           # engine angular speed at max torque [rpm]
gear = [2.666, 1.534, 1.022, 0.721, 0.525, 4.44] # gear ratio
cor_v = [0, 7.5, 11.4, 16, 22] # corresponding velocity for each gear(ex: gear 1 ratio = 2.666,cor_v = 0~7.5) [m/s]

## EV only
# f_ev =                # idle energy consumption[kW]
capacity_kWh = 54       # [kWh]
Voc = 360               # [V]
Pmax_kW = 211           # [kW]
Tmax_Nm = 375           # [Nm]
Tmax_rpm = 4000         # [rpm] max motor speed at max torque
gear_ratio = 9          # motor to driven axle gear ratio
max_speed = 16000       # [rpm] maximum motor speed
dt_eff = 0.85           # drivetrain efficiency
Rcirc = 0.05            # [Ohms]


# Constraint constants to define
pos_final = 2000        # corridor horizon [m]
N = 380                 # time horizon [sec]
vel_lower = 0           # velocity lower bound [m/s]
vel_upper = 17.5        # velocity upper bound [m/s]
acc_abs = 1.5           # accel upper bound [m/s^2]
jerk_abs = 1            # jerk (derivative of acceleration) upper bound [m/s^3]
lin_drag = 10           # linearize drag about this speed

# Traffic light constraints
redlight = 30           # redlight duration [sec]
greenlight = 50         # greenlight duration [sec]
block = 240             # length of one block [m]

# Optimization time horizon
N = 380                 # minimum time is N~=pos_final/10*0.9
dt = 1
M = 1e5


# Helper ==================================================

def set_up(vel_upper=vel_upper, acc_abs=acc_abs, redlight=redlight):
    
    
    # Define optimization vars

    x1 = cp.Variable(N+1) # Position
    x2 = cp.Variable(N+1) # Velocity
    T = cp.Variable(N) # Powertrain torque
    Fdrag = cp.Variable(N) # Vehicle drag
    E = cp.Variable(N+1) # Battery energy
    Pmot = cp.Variable(N) # Battery power
    Vcirc = cp.Variable(N) # Circuit voltage
    I = cp.Variable(N) # Motor current
    SOC = cp.Variable(N+1) # Battery SOC

    # Define objective function
    objective = cp.Minimize( E[0]-E[N] ); title = 'Minimum Energy' # Minimum energy
    # objective = cp.Minimize(-x1@np.ones(N+1) + (E[0]-E[N])); title = 'Minimum Time' # Minimum time - Lower N until solver fails for guaranteed minimum time

    # Define constraints
    constraints = [ ] 
    constraints += [ x1[0] == 0 ] # pos_init = 0 m
    constraints += [ x1[N] == pos_final ] # pos_final = 1000 m
    constraints += [ x2[0] == 0 ] # vel_init = 0 m/s
    constraints += [ x2[N] == 0 ] # vel_final = 0 m/s
    constraints += [ x2[1] == x2[0] ] # # acc_init = 0 m/s^2
    constraints += [ x2[N] == x2[N-1] ] # acc_final = 0 m/s^2
    
    # Traffic light constraints
    # Define a traffic light as [initial time, final time, position]

    tls = []
    for intersection in np.arange(0, pos_final, block):
        for start in np.arange(10, N*dt, redlight + greenlight):
            tls.append([start, start + redlight, intersection]) # + random.randint(-5, 5)

    tls_bools = {}
    tls_on = True # Control whether or not traffic lights are used
    # The for loop below will automatically create constraints for all traffic lights
    for i in range(len(tls)):
        # Ensure that traffic light fits into optimization horizon
        if int(tls[i][0]/dt) <= N and tls_on:            
            dict_index = "tl{0}".format(i)
            tls_bools[dict_index] = cp.Variable(1, boolean=True)
            # Clip end of traffic light if it goes past optimization horizon
            t_initial = int(tls[i][0]/dt)
            if int(tls[i][1]/dt) > N:
                t_final = N
            else:
                t_final = int(tls[i][1]/dt)
            position = tls[i][2]
            constraints += [ x1[t_final]   - position <=  M*tls_bools[dict_index] ]
            constraints += [ x1[t_initial] - position >= -M*(1-tls_bools[dict_index]) ]

    for k in range(0,N):
        
        constraints += [ x2[k+1] >= vel_lower ] # vel_lower >= 0 m/s
        constraints += [ x2[k+1] <= vel_upper ] # vel_lower <= 15 m/s (~55 km/hr)
        constraints += [ (x2[k+1] - x2[k])/dt >= -acc_abs ] # acc_lower >= -3 m/s^2
        constraints += [ (x2[k+1] - x2[k])/dt <= acc_abs ] # acc_upper <= 3 m/s^2
        constraints += [ x1[k+1] == x1[k] + x2[k]*dt ]
        constraints += [ x2[k+1] == x2[k] + ( (T[k])/m/r*gear_ratio*dt_eff - Fdrag[k]/m)*dt ]
        constraints += [ Fdrag[k] >= 0.5*rho*Cd*Aref*x2[k]**2 + Cv*x2[k] + Crr*m*g ]
        constraints += [ E[k+1] == E[k] - Voc*Vcirc[k]/Rcirc*dt/3600 ]
        constraints += [ x2[k]*60/(2*np.pi*r)*gear_ratio <= 10000 ]
        constraints += [ T[k] <= Tmax_Nm/(max_speed-Tmax_rpm)*max_speed + (Tmax_Nm/-(max_speed-Tmax_rpm))*x2[k]*60/(2*np.pi*r)*gear_ratio ]
        constraints += [ T[k] >= - Tmax_Nm/(max_speed-Tmax_rpm)*max_speed - (Tmax_Nm/-(max_speed-Tmax_rpm))*x2[k]*60/(2*np.pi*r)*gear_ratio ]

    # SOC
    constraints += [ SOC[0] == 0.8 ]
    constraints += [ SOC == E/capacity_kWh/1000]
    constraints += [ SOC <= 0.8 ]
    constraints += [ SOC >= 0.2 ]
    # Power
    constraints += [ Pmot <= Vcirc/Rcirc*Voc - (Vcirc)**2/Rcirc ] # Relaxed SOC constraint --> Becomes tight to minimize energy loss
    constraints += [ T <= Tmax_Nm ]
    constraints += [ T >= -Tmax_Nm ]
    constraints += [ T == Pmot/(Pmax_kW*1000)*Tmax_Nm ]
    # Voc
    constraints += [ Vcirc/Rcirc <= (Pmax_kW*1000)/Voc ] # Current limit

    for k in range(1, N):
        # Minimize jerk (derivative of acceleration)
        constraints += [ (x2[k+1] - 2*x2[k] + x2[k-1])/dt**2 <= jerk_abs ]
        constraints += [ (x2[k+1] - 2*x2[k] + x2[k-1])/dt**2 >= -jerk_abs ]
    

    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK')

    E_regen = np.zeros(N+1)
    for i in range(1,len(E_regen)):
        if Vcirc.value[i-1] < 0:
            E_regen[i] = E_regen[i-1] - Voc*Vcirc.value[i-1]/Rcirc*dt/3600
        else:
            E_regen[i] = E_regen[i-1]
    
    var_dict = {
        
        'x1': x1.value,
        'x2': x2.value,
        'T': T.value,
        'Fdrag': Fdrag.value,
        'E': E.value,
        'Pmot': Pmot.value,
        'Vcirc': Vcirc.value,
        'I': I.value,
        'SOC': SOC.value,
        'regen': E_regen
        
    }
    
    return objective, constraints, var_dict

def optimizer_loop(vel_upper_lst=[vel_upper], acc_abs_lst=[acc_abs], redlight_lst=[redlight]):
    
    pos = []
    vel = []
    soc = []
    regen = []
    
    tic_glob = time.perf_counter() 
    
    for vel_upper in vel_upper_lst:
        
        tic = time.perf_counter()
        objective, constraints, var_dict = set_up(vel_upper=vel_upper)
        toc = time.perf_counter()
        
        pos.append(var_dict['x1'])
        vel.append(var_dict['x2'])
        soc.append(var_dict['SOC'])
        regen.append(var_dict['regen'])
        
        print(f"====== Solving with vel_upper = {vel_upper} mph : {toc - tic:0.3f} seconds ======")

    print("")
    for acc_abs in acc_abs_lst:
        
        tic = time.perf_counter()
        objective, constraints, var_dict = set_up(redlight=acc_abs)
        toc = time.perf_counter()
        
        pos.append(var_dict['x1'])
        vel.append(var_dict['x2'])
        soc.append(var_dict['SOC'])
        regen.append(var_dict['regen'])
        
        print(f"====== Solving with acc_abs = {acc_abs} m/s2 : {toc - tic:0.3f} seconds ======")
    
    print("")
    for redlight in redlight_lst:
        
        tic = time.perf_counter()
        objective, constraints, var_dict = set_up(redlight=redlight)
        toc = time.perf_counter()
        
        pos.append(var_dict['x1'])
        vel.append(var_dict['x2'])
        soc.append(var_dict['SOC'])
        regen.append(var_dict['regen'])
    
        print(f"====== Solving with redlight = {redlight} s : {toc - tic:0.3f} seconds ======")
        
    toc_glob = time.perf_counter()
    
    print("")
    print(f"Loop complete")
    print(f"Time used = : {toc_glob - tic_glob:0.3f} seconds")
    
    return pos, vel, soc, regen