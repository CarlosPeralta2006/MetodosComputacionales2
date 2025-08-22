import numpy as np
import matplotlib.pyplot as plt

def generate_data(tmax,dt,A,freq,noise):
     
    #Generates a sin wave of the given amplitude (A) and frequency (freq),
    #sampled at times going from t=0 to t=tmax, taking data each dt units of time.
    #A random number with the given standard deviation (noise) is added to each data point.
    #Returns an array with the times and the measurements of the signal. 
        
     ts = np.arange(0,tmax+dt,dt)
     return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)
 
