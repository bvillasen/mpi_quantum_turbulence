import sys, time, os
import numpy as np
import h5py as h5
from mpi4py import MPI
#Add Modules from other directories
#currentDirectory = os.getcwd()
#parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
#toolsDirectory = parentDirectory + "/tools"
#sys.path.append( toolsDirectory )
#from mpiTools import mpi_id3D

def mpi_id3D( pId, nProc_x, nProc_y ):
  pId_x = pId % nProc_x
  pId_z = pId // (nProc_x*nProc_y)
  pId_y = (pId-pId_z*nProc_y*nProc_x) // nProc_x
  return pId_x, pId_y, pId_z

def densToChar( dens ):
  dens /= dens.max()
  chars = (-255 * ( dens - 1 ) ).astype(np.uint8)
  return chars

def velToChar( vel ):
  vel /= vel.max()
  chars = (-255 * ( vel - 1 ) ).astype(np.uint8)
  return chars

dataDir  = "/home/bruno/Desktop/data/qTurbulence/"
inputDir = "/home_local/bruno/Desktop/data/qTurbulence/snapshots"
fileBase = inputDir + '/psi_'

MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProc = MPIcomm.Get_size()


if pId == 0:
  print "\nMPI compress data"
  print " nProcess: {0}\n".format(nProc) 
    
dimConv = {8:2, 27:3, 64:4, 125:5, 216:6}
nProc_x = dimConv[nProc]
nProc_y = dimConv[nProc]
nProc_z = dimConv[nProc]
if nProc_x*nProc_x*nProc_x !=  nProc:  
  print "ERROR: Number of processes must be an integer cubed (8, 27, 64, 125, 216)" 
  exit()  
  
pId_x, pId_y, pId_z= mpi_id3D( pId, nProc_x, nProc_y )

#Load local data
fileName = fileBase + '{0}_{1}_{2}.h5'.format( pId_z, pId_y, pId_x)
if pId == 0: 
  print "[pId {0}] Input: {1}".format( pId, inputDir )
  print "[pId {0}] Loading data... \n".format( pId )
dataFile = h5.File( fileName ,'r')
fileKeys = dataFile.keys()
nSnaps = len( dataFile.keys() )/5
psiLocal = []
velLocal = []
for i in range( nSnaps ):
  psiKey = str(i) + '_psi_'
  psi = dataFile[psiKey+'R'][...] + 1j*dataFile[psiKey+'I'][...]
  psiLocal.append( psi )
  velKey = str(i) + '_vel_'
  vel_x = dataFile[velKey+'x'][...] 
  vel_y = dataFile[velKey+'y'][...]
  vel_z = dataFile[velKey+'z'][...]
  velLocal.append( np.sqrt( vel_x*vel_x + vel_y*vel_y + vel_z*vel_z ) )
psiLocal = np.array( psiLocal ).astype(np.complex64)
velLocal = np.array( velLocal ).astype(np.float32)
dataShape = psiLocal.shape
dataAll_shape = [ nSnaps, nProc_z*dataShape[1], nProc_y*dataShape[2], nProc_x*dataShape[3] ]
dataFile.close()

if pId == 0: 
  print "[pId {0}] nSnapshots: {1}".format( pId, nSnaps ) 
  print "[pId {0}] Total points: {1} x {2} x {3}\n".format( pId, nProc_z*dataShape[1], nProc_y*dataShape[2], nProc_x*dataShape[3], )

#Process data
densLocal = ( ( np.abs( psiLocal ) )**2 ).astype(np.float32)

#Trasfer data to root
MPIcomm.Barrier()
if pId == 0:
  print "[pId {0}] Transfering data...".format( pId )
  densAll = np.zeros( dataAll_shape, dtype=np.float32 )
  velAll = np.zeros( dataAll_shape, dtype=np.float32 )
  for snap in range( nSnaps ):
    densAll[snap][:dataShape[1], :dataShape[2], :dataShape[3] ] = densLocal[snap]
    velAll [snap][:dataShape[1], :dataShape[2], :dataShape[3] ] = velLocal[snap]
  for idOther in range( 1, nProc ):
    pIdOth_x, pIdOth_y, pIdOth_z= mpi_id3D( idOther, nProc_x, nProc_y )
    densOther = np.zeros_like( densLocal )
    velOther = np.zeros_like( velLocal )
    MPIcomm.Recv(densOther, source=idOther, tag=idOther)
    MPIcomm.Recv(velOther , source=idOther, tag=idOther+1000)
    for snap in range( nSnaps ):
      densAll[snap][pIdOth_z*dataShape[1]:(pIdOth_z+1)*dataShape[1],
		    pIdOth_y*dataShape[2]:(pIdOth_y+1)*dataShape[2],
		    pIdOth_x*dataShape[3]:(pIdOth_x+1)*dataShape[3],] = densOther[snap]
      velAll [snap][pIdOth_z*dataShape[1]:(pIdOth_z+1)*dataShape[1],
		    pIdOth_y*dataShape[2]:(pIdOth_y+1)*dataShape[2],
		    pIdOth_x*dataShape[3]:(pIdOth_x+1)*dataShape[3],] = velOther[snap]
    print "[pId {0}] Recived data from: {1}".format( pId, idOther )
else:  
  MPIcomm.Send( densLocal, dest=0, tag=pId)
  MPIcomm.Send( velLocal , dest=0, tag=pId+1000)

#Save data for volume rendering
if pId == 0:
  print "\n[pId {0}] Saving volume render data".format( pId )
  outputDir = dataDir + "movies/"
  if not os.path.exists( outputDir ): os.makedirs( outputDir )
  densAll = densToChar( densAll )
  velAll  = velToChar( velAll )
  outputFile = outputDir + 'densVolumeRender.h5'
  vrFile = h5.File( outputFile, 'w' )
  for snap in range(nSnaps):
    #if snap%2 == 0:
    snapKey = '{0:03}'.format(snap)
    vrFile.create_dataset( snapKey+'_dens', data=densAll[snap], compression="lzf" )
    vrFile.create_dataset( snapKey+'_vel', data=velAll[snap], compression="lzf" )
  vrFile.close()  
  print "[pId {0}] Saved: {1}".format( pId, outputFile )


######################################################################
#Clean and Finalize
#Terminate MPI
MPIcomm.Barrier()
MPI.Finalize()


#def loadData( snapshot, fileBase ):
  #dataIn = {}
  #for k in range( dim ):
    #for i in range( dim ):
      #for j in range( dim ):
	#fileName = fileBase + '{0}_{1}_{2}.h5'.format( k, i, j)
	#dataFile = h5.File( fileName ,'r')
	#key = '{0}_psi_'.format(snapshot)
	#dataIn[( k, i, j )] = dataFile.get( key + 'R' )[...] + dataFile.get( key + 'I' )[...]*1j
	#dataFile.close()
  #points = dataIn[(0,0,0)].shape[0] 
  #totalPoints = points*dim
  ##print "nPoints: ", totalPoints
  #psiAll  = np.zeros([totalPoints, totalPoints, totalPoints]).astype(np.complex128)
  #psiOutAll = np.zeros([totalPoints, totalPoints, totalPoints])
  #for k in range(dim):
    #for i in range(dim):
      #for j in range(dim):
	#psiAll[k*points:(k+1)*points, i*points:(i+1)*points, j*points:(j+1)*points] = dataIn[(k,i,j)]
	##psiOutAll[i*points:(i+1)*points, j*points:(j+1)*points] = dataOut[(i,j)]
  #return ((np.abs(psiAll))**2).astype( np.float32 )

#def densToChar( dens ):
  #dens /= dens.max()
  #chars = (-255 * ( dens - 1 ) ).astype(np.uint8)
  #return chars


#snapshot = 0
#dataFiles = os.listdir( dataDir + 'snapshots')
#nBlocks = len( [ f for f in dataFiles if f.find('psi')==0 ] )
#dim = int( nBlocks**(1./3)  )

#fileName = fileBase + '0_0_0.h5'
#dataFile = h5.File( fileName ,'r')
#nSnapshots = len( dataFile.keys() ) / 2
#print "\nnSnapshots: {0}".format(nSnapshots)
#print "nBlocks: {0} x {0} x {0}\n".format(dim)




#outputDir = dataDir + "movies/"
#outputFile = outputDir + 'movieData.h5'
#if not os.path.exists( outputDir ): os.makedirs( outputDir )
#movieData = h5.File( outputFile, 'w' )
#for snapshot in range(nSnapshots):
  #snapshot = str( snapshot )
  #print "Saving snapshot: {0}".format( snapshot )
  #data = loadData( snapshot, fileBase )
  #chars = densToChar( data )
  #movieData.create_dataset( snapshot, data=chars, compression="gzip", compression_opts=9 )
#movieData.close() 


