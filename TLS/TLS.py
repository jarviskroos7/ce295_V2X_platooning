import math
import matplotlib.pyplot as plt
import numpy as np
 
# Global parameter adapted from the project code
N = 200

# Local parameters
dt = 1
T = 60
w = 2*(math.pi)/T
h = 0
phase = 0
red_light = []
green_light = []

# Initial state
init = math.sin(w*(0*dt) + phase) + h

# Red light: function >= 0
if init >= 0:
    red_light.append(0)
# Green light: function < 0
else:
    green_light.append(0)

# Traffic light simulation
for i in range(1, N + 1):
    light_after = math.sin(w*(i*dt) + phase) + h
    light_before = math.sin(w*((i-1)*dt) + phase) + h
    # Light changes
    if light_after*light_before < 0:
        # Red light changes to green light
        if light_before >= 0:
            green_light.append(i*dt)
        # Green light changes to red light
        else:
            red_light.append(i*dt)
    # No light change
    else:
        continue

print('Green Light:')
print(green_light)
print('\nRed Light:')
print(red_light)


tls = []

# Light starts in red
if h == 0:
    # Light ends in green
    if len(red_light) == len(green_light):
        for i in range(0, len(green_light)):
            tls.append([red_light[i], green_light[i]])
    # Light ends in red
    else:
        for i in range(0, len(green_light)):
            tls.append([red_light[i], green_light[i]])
        tls.append([red_light[i + 1], N*dt])
# Light starts in red
elif h >= 0:
    # Light ends in green
    if len(red_light) == len(green_light):
        for i in range(0, len(green_light)):
            tls.append([red_light[i], green_light[i]])
    # Light ends in red
    else:
        for i in range(0, len(green_light)):
            tls.append([red_light[i], green_light[i]])
        tls.append([red_light[i + 1], N*dt])
# Light starts in green
else:
    # Light ends in red
    if len(red_light) == len(green_light):
        for i in range(0, len(green_light) - 1):
            tls.append([red_light[i], green_light[i + 1]])
        tls.append([red_light[i + 1], N*dt])
    # Light ends in green
    else:
        for i in range(0, len(green_light) - 1):
            tls.append([red_light[i], green_light[i + 1]])
        
print('\ntls:')
print(tls)
