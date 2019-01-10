import numpy as np
import matplotlib.pyplot as plt
import sys, time, os, datetime
import h5py as h5
import scipy as sci

dataDir = "/home/bruno/Desktop/data/qTurbulence/analysis/"
inFileName = "forward.h5"

inFile = h5.File( dataDir + inFileName, 'r')

proj_r = inFile['projections_r'][...][0]
proj_i = inFile['projections_i'][...][0]
times =  inFile['times'][...]
inFile.close()

proj = np.sqrt( proj_r*proj_r + proj_i*proj_i )
