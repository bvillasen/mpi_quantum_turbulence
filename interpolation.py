import numpy as np #, Image
#import scipy as sci
import matplotlib.pyplot as plt
import sys, time, os, datetime
import h5py as h5
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
 
dataDir = "/home/bruno/Desktop/data/qTurbulence/initial/"
inFileName = 'converged/psiConverged_10_1e-4.h5' 

#Load converged psi
print 'Loading input: {0}'.format( dataDir + inFileName )
h5_in = h5.File( dataDir + inFileName, 'r')
#h5_in = h5.File( dataDir + 'initial/converged/psiConverged_128_sph.h5', 'r')
psiConverged = h5_in.get('psi')[...]
h5_in.close()
nDepth, nHeight, nWidth = psiConverged.shape

Lx = 30.0
Ly = 30.0
Lz = 30.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
xPoints = dx * np.arange(nWidth)  + xMin
yPoints = dy * np.arange(nHeight) + yMin
zPoints = dz * np.arange(nDepth)  + zMin

#Interpolated grid
nPoints = 128*3
nDepth_i, nHeight_i, nWidth_i = nPoints, nPoints, nPoints
#zFartor, yFactor, xFactor = 6, 6, 6
#nDepth_i, nHeight_i, nWidth_i = nDepth*zFartor, nHeight*yFactor, nWidth*xFactor
print '\nInitial  shape: {0} x {1} x {2}'.format( nDepth, nHeight, nWidth )
print 'Interpol shape: {0} x {1} x {2}'.format( nDepth_i, nHeight_i, nWidth_i )

psiInterpol = np.zeros( [nDepth_i, nHeight_i, nWidth_i ], dtype=complex ) 
psiInterpol_partial = np.zeros( [nDepth_i, nHeight, nWidth ], dtype=complex )
dx_i, dy_i, dz_i = Lx/(nWidth_i-1), Ly/(nHeight_i-1), Lz/(nDepth_i-1 )
xPoints_i = dx_i * np.arange(nWidth_i)  + xMin
yPoints_i = dy_i * np.arange(nHeight_i) + yMin
zPoints_i = dz_i * np.arange(nDepth_i)  + zMin
nPoints = np.array([ nDepth_i, nHeight_i, nWidth_i ])

print '\n1D interpolation across z-axis...'
start = time.time()
for i in range( nHeight ):
  for j in range( nWidth ):
    lineR = psiConverged[:,i,j].real
    lineI = psiConverged[:,i,j].imag
    splineR = UnivariateSpline(zPoints, lineR, k=5, s=0 )
    splineI = UnivariateSpline(zPoints, lineI, k=5, s=0 )
    psiInterpol_partial[:,i,j].real = splineR( zPoints_i )
    psiInterpol_partial[:,i,j].imag = splineI( zPoints_i )
print " Time: ", time.time() - start

print '\n2D interpolation transverse to z-axis...'
start = time.time()
for k in range( nDepth_i ):
  splineR = RectBivariateSpline( xPoints, yPoints, psiInterpol_partial[k,:,:].real, kx=5, ky=5, s=0 )
  splineI = RectBivariateSpline( xPoints, yPoints, psiInterpol_partial[k,:,:].imag, kx=5, ky=5, s=0 )
  psiInterpol[k,:,:].real = splineR( xPoints_i, yPoints_i )
  psiInterpol[k,:,:].imag = splineI( xPoints_i, yPoints_i )
print " Time: ", time.time() - start

#Output Data
outDir = dataDir + 'interpol/'
if not os.path.exists( outDir ): os.makedirs( outDir )
outFileName = outDir + 'psi_{0}.h5'.format( nDepth_i )
print "\nSaving data: {0}".format( outFileName )
start = time.time()
h5_out = h5.File( outFileName, "w")
h5_out.create_dataset("nPoints", data=nPoints  )
h5_out.create_dataset("psi", data=psiInterpol )
#h5_out.create_dataset("psiI", data=psiInterpol.imag )
#psiHD.attrs['nPoints'] = np.array([ nDepth_i, nHeight_i, nWidth_i ]) 
h5_out.close()
print " Time: ", time.time() - start

























#line = psiConverged[:,128,128].real
#line_i = psiInterpol[:,256,256].real
#plt.figure(0)
#plt.clf()
#plt.plot( zPoints_i, line_i, 'go' )
#plt.plot( zPoints, line, 'bo' )

#plt.figure(1)
#plt.clf()
#plt.imshow( psiInterpol[k,::yFactor,::xFactor].real, interpolation='nearest' )
#plt.figure(2)
#plt.clf()
#plt.imshow( psiInterpol[k,:,:].real, interpolation='nearest' )
#plt.show()











#x = np.arange(0, 10)
#y = np.exp(-x/3.0)
#f = interpolate.interp1d(x, y, kind=8)

#xnew = np.arange(0,9, 0.1)
#ynew = f(xnew)   # use interpolation function returned by `interp1d`
#plt.clf()
#plt.plot(x, y, 'ro', xnew, ynew, 'b-')
#plt.plot(xnew, np.exp(-xnew/3.0), 'g-' )
#plt.show()


#from numpy import linspace,exp
#from numpy.random import randn
#import matplotlib.pyplot as plt

#x = linspace(-3, 3, 100)
#y = exp(-x**2) + randn(100)/10
#s = UnivariateSpline(x, y, s=1)
#xs = linspace(-3, 3, 1000)
#ys = s(xs)
#plt.plot(x, y, '.-')
#plt.plot(xs, ys)
#plt.show()