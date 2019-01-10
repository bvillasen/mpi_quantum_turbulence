import sys, time, os
import numpy as np
import h5py as h5
from mpi4py import MPI



nPoints = 128 * 3
clearData = False
#Read in-line parameters
for option in sys.argv:
  if option == 'clear' : clearData = True
  
dataDir = "/home/bruno/Desktop/data/qTurbulence/"
initDir = "/home_local/bruno/Desktop/data/qTurbulence/initial/"
snapDir = "/home_local/bruno/Desktop/data/qTurbulence/snapshots/"
if not os.path.exists( initDir ): os.makedirs( initDir )
if not os.path.exists( snapDir ): os.makedirs( snapDir )
#inFileName = dataDir + 'initial/converged/psiConverged_10_1e-4.h5'
inFileName = dataDir + 'initial/interpol/psi_{0}.h5'.format( nPoints )

MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProc = MPIcomm.Get_size()

if clearData:
  if pId < 20:
    for direc in [ initDir, snapDir ]:
      fileList = os.listdir( direc )
      for fileName in fileList:
	os.remove( direc + fileName )
  sys.exit()
  
if pId == 0:
  print "\nUsing MPI"
  print " nProcess: {0}".format(nProc) 
  print '\nInput: {0}\n'.format( inFileName )
    
dimConv = {8:2, 27:3, 64:4, 125:5, 216:6}
nProc_x = dimConv[nProc]
nProc_y = dimConv[nProc]
nProc_z = dimConv[nProc]
if nProc_x*nProc_x*nProc_x !=  nProc:  
  print "ERROR: Number of processes must be an integer cubed (8, 27, 64, 125, 216)" 
  exit()  
pId_x = pId % nProc_x
pId_z = pId // (nProc_x*nProc_y)
pId_y = (pId-pId_z*nProc_y*nProc_x) // nProc_x

Lx = 30.0
Ly = 30.0
Lz = 30.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2

#if pId == 0:
  ##Load converged psi
  #inFile = h5.File( inFileName, 'r')
  #inShape = inFile.get('nPoints')
  #nDepth, nHeight, nWidth = psiConverged.attrs[ 'nPoints' ]
#MPIcomm.Barrier()
#inShape = MPIcomm.Bcast(inShape, root=0) 

nDepth, nHeight, nWidth = nPoints, nPoints, nPoints 
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
xPoints = dx * np.arange(nWidth)  + xMin
yPoints = dy * np.arange(nHeight) + yMin
zPoints = dz * np.arange(nDepth)  + zMin

nDepth_p, nHeight_p, nWidth_p = nDepth/nProc_z, nHeight/nProc_y, nWidth/nProc_x 
nData_p = nDepth_p * nHeight_p * nWidth_p
xPoints_p = xPoints[ pId_x*nWidth_p:  (pId_x+1)*nWidth_p ]
yPoints_p = yPoints[ pId_y*nHeight_p: (pId_y+1)*nHeight_p ]
zPoints_p = zPoints[ pId_z*nDepth_p:  (pId_z+1)*nDepth_p ]
boxRange  = np.array([ [xPoints_p[0], xPoints_p[-1]], 
		      [yPoints_p[0], yPoints_p[-1]], 
		      [zPoints_p[0], zPoints_p[-1]] ])

psiLocal = np.zeros( nData_p, dtype=np.complex128 )
		    
if pId == 0:
  print '[pID:{3}] Total points: ( {0} x {1} x {2} )'.format(  nDepth, nHeight, nWidth, pId )
  print '[pID:{3}] Points per core: ( {0} x {1} x {2} )\n'.format(  nDepth_p, nHeight_p, nWidth_p, pId )

  #Load interpolated psi
  print '[pID:{0}] Loading data...'.format(pId)
  inFile = h5.File( inFileName, 'r')
  psiConverged = inFile.get('psi')[...]
  psiLocal = psiConverged[0:nDepth_p, 0:nHeight_p, 0:nWidth_p].reshape(nData_p)
  for pIdOth in range(1, nProc):
    pIdOth_x = pIdOth % nProc_x
    pIdOth_z = pIdOth // (nProc_x*nProc_y)
    pIdOth_y = (pIdOth-pIdOth_z*nProc_y*nProc_x) // nProc_x
    psiOth = psiConverged[ pIdOth_z*nDepth_p  : (pIdOth_z+1)*nDepth_p,
	     		   pIdOth_y*nHeight_p : (pIdOth_y+1)*nHeight_p,
			   pIdOth_x*nWidth_p  : (pIdOth_x+1)*nWidth_p ]
    psiOth = psiOth.reshape(nData_p)
    print '[pID:{0}] Sending data to: {1}'.format(pId, pIdOth)
    MPIcomm.Send( psiOth, dest=pIdOth, tag=pIdOth)

else: MPIcomm.Recv(psiLocal, source=0, tag=pId)

psiLocal = psiLocal.reshape( nDepth_p, nHeight_p, nWidth_p )








  #if pId == i: print '[pID:{0}] ({1},{2},{3}) [[{4:.2f},{5:.2f}] [{6:.2f},{7:.2f}] [{8:.2f},{9:.2f}]]'.format( pId, pId_z, pId_y, pId_x, boxRange[0,0], boxRange[0,1], boxRange[1,0], boxRange[1,1], boxRange[2,0], boxRange[2,1] )
  #time.sleep(0.2)
  #MPIcomm.Barrier()
#
#dataOut = psiConverged[ pId_z*nDepth_p  : (pId_z+1)*nDepth_p,
			#pId_y*nHeight_p : (pId_y+1)*nHeight_p,
			#pId_x*nWidth_p  : (pId_x+1)*nWidth_p ]
  


FileBase = initDir + 'psi_in_'
outputName = FileBase + '{0}_{1}_{2}.h5'.format( pId_z, pId_y, pId_x)
if pId == 0: print "\n[pID: {0}] Saving data: {1}".format(pId, outputName)
outFile = h5.File( outputName ,'w')
outFile.create_dataset( 'boxRange', data = boxRange )
outFile.create_dataset( 'dX', data = np.array([ dx, dy, dz ]) )
outFile.create_dataset( 'xPoints', data = xPoints_p )
outFile.create_dataset( 'yPoints', data = yPoints_p )
outFile.create_dataset( 'zPoints', data = zPoints_p )
outFile.create_dataset( "psi_R", data=psiLocal.real )
outFile.create_dataset( "psi_I", data=psiLocal.imag )
outFile.close()


######################################################################
#Clean and Finalize
if pId == 0: inFile.close()

#Terminate MPI
MPIcomm.Barrier()
MPI.Finalize()

