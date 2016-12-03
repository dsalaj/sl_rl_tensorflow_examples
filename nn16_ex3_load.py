import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
from numpy import *    

pl.close('all')   # closes all previous figures

# Load dataset
file_in = open('vehicle.pkl','rb')
vehicle_data = pckl.load(file_in)
file_in.close()
X = vehicle_data[0]   # input vectors X[i,:] is i-th example
C = vehicle_data[1]   # classes C[i] is class of i-th example

    # Remove unneeded data
    # Classify the rest 'manually' (no tensor flow)
