import pandas as pd
import os
import numpy as np
import IPython

DESCRIPTION = 'Use nominal measured values for each sphere drop to compute viscosity using Stokes Law. Estimate Reynolds numbers'

def Stokes(arg):
    """Function to calculate viscosity using Stokes law. Also calculates Reynolds number and returns whether or not Stoke's law is applicable."""
    d = arg[0]/1000#sphere diameter, m
    m = arg[1]/1000 #mass of sphere, kg
    t = arg[2] #time to fall, s
    h = arg[3] #distance traveled, m
    SG = arg[4]
    r = d/2 #radius of sphere, m
    volume_sphere = (4/3)*np.pi*(r**3)
    p_H20 = 1000 #density of water kg/m^3
    g = 9.81 #gravitational acceleration, m/s^2
    p_sphere = m/volume_sphere#density of sphere kg/m^3
    p_liquid = SG*p_H20 #density of fluid kg/m^3
    V = h/t #velocity of sphere, m/s

    viscosity = (d**2)*g*(p_sphere-p_liquid)/(V*18) #Viscosity in kg/(m*s)
    Reynolds = p_liquid*V*d/(viscosity)
    if Reynolds < 0.1:
        applic = 1
    else:
        applic = 0
    return(np.round(viscosity,3),np.round(Reynolds,3),applic)


#Read Data from file
fileToRead = 'spheredrop.xlsx'
mess = pd.read_excel('spheredrop.xlsx', skiprows=[0,1,3])
mess = mess.dropna()
#Parse out relevant Data

d = np.array(mess['d'])
m = np.array(mess['m'])
t = np.array(mess['t'])
h = np.array(mess['h'])
SG = np.array(mess['S'])
data =np.array([d,m,t,h,SG]).T

#Apply stokes law to find viscosity
for i, trial in enumerate(data):
    print('Trial '+str(i) +':', Stokes(trial))
