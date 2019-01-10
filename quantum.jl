using MPI
using HDF5
# using PyPlot
# push!(LOAD_PATH, "/Path/To/My/Module/")
using tools

MPI.Init()
comm = MPI.COMM_WORLD
const pId = MPI.Comm_rank(comm)
const nProc = MPI.Comm_size(comm)
const root = 0

if pId == root 
  print( "\nQuantum 3D\n\nUsing MPI\nnProcess: $nProc\n" ) 
end

dataDir = "/home/bruno/Desktop/data/qTurbulence/"
initDir = "/home_local/bruno/Desktop/data/qTurbulence/initial/"
snapDir = "/home_local/bruno/Desktop/data/qTurbulence/snapshots/"

##################################################################
const nProc_x = int( nProc^(1/3))
const nProc_y = int( nProc^(1/3))
const nProc_z = int( nProc^(1/3))
if nProc_x*nProc_x*nProc_x !=  nProc  
  error("ERROR: Number of processes must be an integer cubed (8, 27, 64, 125, 216") 
  exit()
end
const pId_z = divrem( pId, nProc_x*nProc_y)[1] #pId // (nProc_x*nProc_y)
const pId_x = divrem( pId, nProc_x )[2]  #pId % nProc_x
const pId_y = divrem( pId - pId_z*nProc_y*nProc_z, nProc_x )[1]  #(pId-pId_z*nProc_y*nProc_x) // nProc_x
const pParity = mod( pId_x + pId_y + pId_z, 2 )
#Boundery processes and periodic Boundery processes
const pDownId   = pId_y == 0         ? pId_z*nProc_x*nProc_y + (nProc_y-1)*nProc_x + pId_x : pId_z*nProc_x*nProc_y + (pId_y-1)*nProc_x + pId_x
const pUpId     = pId_y == nProc_y-1 ? pId_z*nProc_x*nProc_y + pId_x                       : pId_z*nProc_x*nProc_y + (pId_y+1)*nProc_x + pId_x
const pLeftId   = pId_x == 0         ? pId_z*nProc_x*nProc_y + pId_y*nProc_x + nProc_x-1   : pId_z*nProc_x*nProc_y + pId_y*nProc_x + (pId_x-1)
const pRightId  = pId_x == nProc_x-1 ? pId_z*nProc_x*nProc_y + pId_y*nProc_x               : pId_z*nProc_x*nProc_y + pId_y*nProc_x + (pId_x+1)
const pBottomId = pId_z == 0         ? (nProc_z-1)*nProc_x*nProc_y + pId_y*nProc_x + pId_x : (pId_z-1)*nProc_x*nProc_y + pId_y*nProc_x + pId_x
const pTopId    = pId_z == nProc_z-1 ? pId_y*nProc_x + pId_x                               : (pId_z+1)*nProc_x*nProc_y + pId_y*nProc_x + pId_x
##################################################################

initialFile = "psi_in_$(pId_z)_$(pId_y)_$(pId_x).h5"
initialDataName = initDir *  initialFile

if pId == root
  print( "\n[ pID: $pId ] Loading initial state: " * initialFile * "\n" )
end
initialStateFile = h5open( initialDataName, "r")
dx_p, dy_p, dz_p = read( initialStateFile["dX"] )
xPoints = read( initialStateFile["xPoints"] )
yPoints = read( initialStateFile["yPoints"] )
zPoints = read( initialStateFile["zPoints"] )
xMin_p, yMin_p, zMin_p = xPoints[1], yPoints[1], zPoints[1]
xMax_p, yMax_p, zMax_p = xPoints[end], yPoints[end], zPoints[end]
# print( "pID: $pId, ( $pId_z , $pId_y , $pId_x ) : [ ( $zMin_p, $zMax_p ) ( $yMin_p, $yMax_p ) ( $xMin_p, $xMax_p )  ]\n" )
##################################################################

const nIterations = 160000
const nPartialSteps = 100

const dt = 7.5e-4

const timeDirection = "forward"
# loadState = false

const nWidth_p  = length( xPoints )
const nHeight_p = length( yPoints )
const nDepth_p  = length( zPoints )
const xMin = xMin_p
const yMin = yMin_p
const zMin = zMin_p
const dx = dx_p
const dy = dy_p
const dz = dz_p
const dxInv = 1/dx
const dyInv = 1/dy
const dzInv = 1/dz
const dxInv2 = 1/(dx*dx)
const dyInv2 = 1/(dy*dy)
const dzInv2 = 1/(dz*dz)
const nPoints = nWidth_p * nProc_x

const omega = 0
const g = 8000.
const gammaX = 1
const gammaY = 1
const gammaZ = 1

if pId == root
  print( "\n[ pID: $pId ] Blocks: $(nProc_x) x $(nProc_y) x $(nProc_z)\n" )
  print(   "[ pID: $pId ] Points: $(nProc_x*nWidth_p) x $(nProc_y*nHeight_p) x $(nProc_z*nDepth_p)\n" )
  print(   "[ pID: $pId ] PpB:    $(nWidth_p) x $(nHeight_p) x $(nDepth_p)\n\n" )
end
##################################################################

initialState_R = read( initialStateFile["psi_R"] )
initialState_I = read( initialStateFile["psi_I"] )
close( initialStateFile )

#Initialize data
const nNeig = 1
localSum_r = zeros( Float64, nPartialSteps +2)
localSum_i = zeros( Float64, nPartialSteps +2)
timeSamples = zeros( Float64, nPartialSteps +2)
psi = ones( Complex128, (nDepth_p+2*nNeig, nHeight_p+2*nNeig, nWidth_p+2*nNeig) )
# psiInit = ones( Complex128, (nDepth_p+2*nNeig, nHeight_p+2*nNeig, nWidth_p+2*nNeig) )

radius = 0.0796842105263 
blockSum = 0.0
for k in 1:nDepth_p
  z = (k-1)*dz + zMin
  for i in 1:nHeight_p
    y = (i-1)*dy + yMin
    for j in 1:nWidth_p
      x = (j-1)*dx + xMin    
      psiVal = initialState_R[ j, i, k ] + initialState_I[ j, i, k ]*1im
      #Z vortex
      x0, y0 = -1.3625, 0.
      Q = 1
      vtxR = sqrt( ( ( x - x0 )^2 + ( y - y0 )^2 )/radius )
      vtxDens = tanh( vtxR^( 2 * abs(Q) ) )
      vtxPhase = exp( 1im * Q * atan2( x-x0, y-y0 ) )
      psiVal *= ( vtxPhase * vtxDens )
      #Z vortex
      x0, y0 = 1.3625, 0.
      Q = -1
      vtxR = sqrt( ( ( x - x0 )^2 + ( y - y0 )^2 )/radius )
      vtxDens = tanh( vtxR^( 2 * abs(Q) ) )
      vtxPhase = exp( 1im * Q * atan2( x-x0, y-y0 ) )
      psiVal *= ( vtxPhase * vtxDens )
 
#       #Y vortex
#       x0, z0 = -40*dx, -0*dz
#       Q = 1
#       vtxR = sqrt( ( ( x - x0 )^2 + ( z - z0 )^2 )/radius )
#       vtxDens = tanh( vtxR^( 2 * abs(Q) ) )
#       vtxPhase = exp( 1im * Q * atan2( x-x0, z-z0 ) )
#       psiVal *= ( vtxPhase * vtxDens )
      #############################################################
      psi[ j+nNeig, i+nNeig, k+nNeig ] = psiVal
      blockSum += ( abs2( psiVal ) * dx*dy*dz )
    end
  end
end

MPI.Barrier(comm)

#Normalize
sumValue = MPI.Reduce(blockSum, MPI.SUM, root, comm)
totalSum = [ 0.0 ]
if pId == root
  totalSum[1] = sumValue 
  #Bradcast total blockSum
  for id = 1 : (nProc-1)
    MPI.Send( totalSum, id, 1, comm )
  end
#   print( "Norm: $(sum)\n" )
else 
  MPI.Recv!( totalSum, root, 1, comm )
end
# totalSum = MPI.bcast(totalSum, root, comm)
# print( "[ pID: $pId ] Norm: $(totalSum[1]) \n" )  
psi /= totalSum[1]

if timeDirection == "backward"
  fileName = "finalPsi_$(pId_z)_$(pId_y)_$(pId_x).h5"
  if pId == root
    print( "\n[ pID: $pId ] Loading initial state: " * fileName * "\n\n" )
  end
  initialStateFile = h5open( snapDir * fileName, "r")
  initialState_R = read( initialStateFile["0_psi_R"] )
  initialState_I = read( initialStateFile["0_psi_I"] )
  close( initialStateFile )
  psi[nNeig+1:end-nNeig, nNeig+1:end-nNeig, nNeig+1:end-nNeig] = initialState_R + 1im*initialState_I
  conj!(psi)
end 

const psiInit = copy( psi )
psiK1 = copy(psi)
psiK2 = copy(psi)
psiRunge = copy(psi)
velocity_x = zeros( Float32, ( nDepth_p, nHeight_p, nWidth_p ) )
velocity_y = zeros( Float32, ( nDepth_p, nHeight_p, nWidth_p ) )
velocity_z = zeros( Float32, ( nDepth_p, nHeight_p, nWidth_p ) )
##################################################################
bounderyUp_new     = zeros( Complex128, ( nDepth_p+2,  nWidth_p+2 ) )
bounderyDown_new   = zeros( Complex128, ( nDepth_p+2,  nWidth_p+2 ) )
bounderyLeft_new   = zeros( Complex128, ( nHeight_p+2, nDepth_p+2 ) )
bounderyRight_new  = zeros( Complex128, ( nHeight_p+2, nDepth_p+2 ) )
bounderyTop_new    = zeros( Complex128, ( nHeight_p+2, nWidth_p+2 ) )
bounderyBottom_new = zeros( Complex128, ( nHeight_p+2, nWidth_p+2 ) )
##################################################################
function transferBounderies( psiIn::Array{Complex128} )
  bounderyRight  = psiIn[ nWidth_p+1,           :,          : ]
  bounderyLeft   = psiIn[          2,           :,          : ]
  bounderyUp     = psiIn[          :, nHeight_p+1,          : ]
  bounderyDown   = psiIn[          :,           2,          : ]
  bounderyTop    = psiIn[          :,           :, nDepth_p+1 ]
  bounderyBottom = psiIn[          :,           :,          2 ]
  
#   MPI.Barrier(comm)
  if pParity == 0
  
    MPI.Send( bounderyLeft, pLeftId, 3, comm )
    MPI.Recv!( bounderyLeft_new, pRightId, 3, comm )
    psiIn[ nWidth_p+2, :, : ] = bounderyLeft_new
    
    MPI.Send( bounderyRight, pRightId, 4, comm )
    MPI.Recv!( bounderyRight_new, pLeftId, 4, comm )
    psiIn[ 1, :, : ] = bounderyRight_new
    
    MPI.Send( bounderyUp, pUpId, 1, comm )
    MPI.Recv!( bounderyUp_new, pDownId, 1, comm )
    psiIn[ :, 1, :] = bounderyUp_new
    
    MPI.Send( bounderyDown, pDownId, 2, comm )
    MPI.Recv!( bounderyDown_new, pUpId, 2, comm )
    psiIn[ :, nHeight_p+2, :] = bounderyDown_new
        
    MPI.Send( bounderyTop, pTopId, 5, comm )
    MPI.Recv!( bounderyTop_new, pBottomId, 5, comm )
    psiIn[ :, :, 1] = bounderyTop_new
    
    MPI.Send( bounderyBottom, pBottomId, 6, comm )
    MPI.Recv!( bounderyBottom_new, pTopId, 6, comm )
    psiIn[ :, :, nDepth_p+2] = bounderyBottom_new
  else
  
    MPI.Recv!( bounderyLeft_new, pRightId, 3, comm )
    MPI.Send( bounderyLeft, pLeftId, 3, comm )
    psiIn[ nWidth_p+2, :, : ] = bounderyLeft_new
    
    MPI.Recv!( bounderyRight_new, pLeftId, 4, comm )
    MPI.Send( bounderyRight, pRightId, 4, comm )
    psiIn[ 1, :, : ] = bounderyRight_new
      
    MPI.Recv!( bounderyUp_new, pDownId, 1, comm )
    MPI.Send( bounderyUp, pUpId, 1, comm )
    psiIn[ :, 1, :] = bounderyUp_new
    
    MPI.Recv!( bounderyDown_new, pUpId, 2, comm )
    MPI.Send( bounderyDown, pDownId, 2, comm )
    psiIn[ :, nHeight_p+2, :] = bounderyDown_new
            
    MPI.Recv!( bounderyTop_new, pBottomId, 5, comm )
    MPI.Send( bounderyTop, pTopId, 5, comm )
    psiIn[ :, :, 1] = bounderyTop_new
    
    MPI.Recv!( bounderyBottom_new, pTopId, 6, comm )
    MPI.Send( bounderyBottom, pBottomId, 6, comm )
    psiIn[ :, :, nDepth_p+2] = bounderyBottom_new
  end
  MPI.Barrier(comm)
end

##################################################################
function core( j::Int, i::Int, k::Int, x::Float64, y::Float64, z::Float64,
               psiIn::Array{Complex128}  )
  c  = psiIn[   j,   i,     k ]
  r  = psiIn[ j+1,   i,     k ]
  l  = psiIn[ j-1,   i,     k ]
  u  = psiIn[   j, i+1,     k ]
  d  = psiIn[   j, i-1,     k ]
  t  = psiIn[   j,   i,   k+1 ]
  b  = psiIn[   j,   i,   k-1 ]
  
  lap  = ( l + r - 2c )*dxInv2 
  lap += ( d + u - 2c )*dyInv2 
  lap += ( t + b - 2c )*dzInv2

#     derv_x = ( r - l )/(2*dx)
#     derv_y = ( u - d )/(2*dy)
#     Lz = 1im*( derv_y*x - derv_x*y )*omega

  GP = g * abs2( c ) 
#   vTrap = 0.5 * ( gammaX*x*x + gammaY*y*y + gammaZ*z*z )
  vTrap = 0.5 * ( x*x + y*y + z*z )
  
  return 1im * ( 0.5*lap - (vTrap + GP)*c  )
end
##################################################################
function eulerStepCenter( weight::Float64, slopeCoef::Float64, lastRK4step::Bool, 
	psi::Array{Complex128}, psiIn::Array{Complex128}, 
	psiOut::Array{Complex128}, psiRunge::Array{Complex128})

  for k = 2:nDepth_p+1 
    z = (k-2)*dz + zMin
    for i = 2:nHeight_p+1
      y = (i-2)*dy + yMin
      for j = 2:nWidth_p+1
	x = (j-2)*dx + xMin
	value = dt * core( j, i, k, x, y, z, psiIn )
	if lastRK4step
	  value = psiRunge[j, i, k] + slopeCoef*value
	  psi[ j, i, k ]      = value
	  psiOut[ j, i, k ]   = value
	  psiRunge[ j, i, k ] = value
	else
	  psiOut[ j, i, k ]   = psi[ j, i, k ]      + weight*value
	  psiRunge[ j, i, k ] = psiRunge[ j, i, k ] + slopeCoef*value
	end
      end
    end
  end
end

##################################################################
function transfer_compute( weight::Float64, slopeCoef::Float64, lastRK4step::Bool, 
	psi::Array{Complex128}, psiIn::Array{Complex128}, 
	psiOut::Array{Complex128}, psiRunge::Array{Complex128} )

  transferTime = @elapsed transferBounderies( psiIn )
  computeTime  = @elapsed eulerStepCenter( weight, slopeCoef, lastRK4step, psi, psiIn, psiOut, psiRunge )

  return [ computeTime transferTime 0. ]
end

##################################################################
function rk4step(psiIn::Array{Complex128}, psiK1::Array{Complex128},
	      psiK2::Array{Complex128}, psiRunge::Array{Complex128} )

  time = [ 0. 0. 0. ]
  
  #Step 1
  slopeCoef = 1.0/6
  weight = 0.5
  time += transfer_compute( weight, slopeCoef, false, psi, psiK2, psiK1, psiRunge )

  #Step 2
  slopeCoef = 2.0/6
  weight = 0.5
  time += transfer_compute( weight, slopeCoef, false, psi, psiK1, psiK2, psiRunge )

  #Step 3
  slopeCoef = 2.0/6
  weight = 1.0
  time += transfer_compute( weight, slopeCoef, false, psi, psiK2, psiK1, psiRunge )

  #Step 4
  slopeCoef = 1.0/6
  weight = 1.0
  time += transfer_compute( weight, slopeCoef, true, psi, psiK1, psiK2, psiRunge )

  return time
end
##################################################################
function velocity_core( j::Int, i::Int, k::Int, psiIn::Array{Complex128}  )
  c  = psiIn[   j,   i,     k ]
  r  = psiIn[ j+1,   i,     k ]
  l  = psiIn[ j-1,   i,     k ]
  u  = psiIn[   j, i+1,     k ]
  d  = psiIn[   j, i-1,     k ]
  t  = psiIn[   j,   i,   k+1 ]
  b  = psiIn[   j,   i,   k-1 ]
  gradient_x = ( r - l)*dxInv*0.5
  gradient_y = ( u - d)*dyInv*0.5
  gradient_z = ( t - b)*dzInv*0.5
  
  rho = abs2(c) + 5e-6
  vel_x = ( real(c) * imag(gradient_x) - imag(c) * real(gradient_x) ) / rho
  vel_y = ( real(c) * imag(gradient_y) - imag(c) * real(gradient_y) ) / rho
  vel_z = ( real(c) * imag(gradient_z) - imag(c) * real(gradient_z) ) / rho
  
  return [ vel_x vel_y vel_z ]
end
##################################################################
function getVelocity( psiIn::Array{Complex128}, velocity_x::Array{Float32},
                      velocity_y::Array{Float32}, velocity_z::Array{Float32} )
  for k = 2:nDepth_p+1 
    for i = 2:nHeight_p+1
      for j = 2:nWidth_p+1
	vel = velocity_core( j, i, k, psiIn)
	velocity_x[ j-1, i-1, k-1 ] = float32( vel[1] )
	velocity_y[ j-1, i-1, k-1 ] = float32( vel[2] )
	velocity_z[ j-1, i-1, k-1 ] = float32( vel[3] )
      end
    end
  end
end
##################################################################
function integratePartial( contador::Int, nIterations::Int, 
                    psiIn::Array{Complex128}, psiK1::Array{Complex128}, 
                    psiK2::Array{Complex128}, psiRunge::Array{Complex128},
                    localSum_r::Array{Float64},localSum_i::Array{Float64},
                    psiInit::Array{Complex128})
  time = [ 0. 0. 0. ]
  for n = 1:nIterations
    time +=  rk4step( psiIn, psiK1, psiK2, psiRunge )
  end
  projection = sum(psiInit[2:end-1, 2:end-1, 2:end-1] .* conj(psiIn[2:end-1, 2:end-1, 2:end-1]) )*dx*dy*dz
  localSum_r[contador] = real( projection )
  localSum_i[contador] = imag( projection )
  
  return time 
end

function saveState_32( snapshot, outputFile )
  nRes = 3
  outputFile["$(snapshot)_psi_R"] = float32( real( psi[2:nRes:end-1, 2:nRes:end-1, 2:nRes:end-1] ) )
  outputFile["$(snapshot)_psi_I"] = float32( imag( psi[2:nRes:end-1, 2:nRes:end-1, 2:nRes:end-1] ) )
  outputFile["$(snapshot)_vel_x"] =   velocity_x[1:nRes:end, 1:nRes:end, 1:nRes:end]  
  outputFile["$(snapshot)_vel_y"] =   velocity_y[1:nRes:end, 1:nRes:end, 1:nRes:end]  
  outputFile["$(snapshot)_vel_z"] =   velocity_z[1:nRes:end, 1:nRes:end, 1:nRes:end]  
  
end
function saveState_64( snapshot, outputFile )
  nRes = 1
  outputFile["$(snapshot)_psi_R"] = real( psi[2:nRes:end-1, 2:nRes:end-1, 2:nRes:end-1] ) 
  outputFile["$(snapshot)_psi_I"] = imag( psi[2:nRes:end-1, 2:nRes:end-1, 2:nRes:end-1] ) 
end
##################################################################
MPI.Barrier(comm)

fileName = "psi_$(pId_z)_$(pId_y)_$(pId_x).h5"
snapshotsFile = h5open( snapDir * fileName, "w")



transferBounderies( psi )
getVelocity( psi, velocity_x, velocity_y, velocity_z )
saveState_32( 0,  snapshotsFile )

MPI.Barrier(comm)
if pId == root
  print( "[ pID: $(pId) ] Starting $nIterations iterations\n" ) 
  print( "[ pID: $(pId) ] Simulation time: $(nIterations*dt) \n\n" ) 
end

nIterations_partial, reminderSteps = divrem( nIterations, nPartialSteps )   
totalTime = [ 0. 0. 0. ] 
projection = sum(psiInit[2:end-1, 2:end-1, 2:end-1] .* conj(psi[2:end-1, 2:end-1, 2:end-1])  )*dx*dy*dz
localSum_r[1] = real( projection )
localSum_i[1] = imag( projection )
count = 2
simulationTime = 0
for i in 0:nPartialSteps
  if pId == 0 
    time = totalTime[1] + totalTime[2] + totalTime[3]
    tools.printProgress( i, nPartialSteps, time )
  end
  totalTime += integratePartial( count, nIterations_partial, psi, psiK1, psiK2, psiRunge, localSum_r, localSum_i, psiInit )
  totalTime[2] += @elapsed transferBounderies( psi )
  totalTime[1] += @elapsed getVelocity( psi, velocity_x, velocity_y, velocity_z )
  totalTime[3] += @elapsed saveState_32( i+1, snapshotsFile )
  simulationTime += nIterations_partial * dt
  timeSamples[ count ] = simulationTime
  count += 1
end

MPI.Barrier(comm)

#Sum all projections 
# sum = MPI.Reduce(blockSum, MPI.SUM, root, comm)
projectionSum_r = zeros( Float64, ( size( localSum_r )[1], nProc, ) )
projectionSum_i = zeros( Float64, ( size( localSum_i )[1], nProc, ) )

localSumOther_r = zeros( Float64,  size( localSum_r ) )
localSumOther_i = zeros( Float64,  size( localSum_i ) )

if pId == root
  projectionSum_r[:,1] = localSum_r 
  projectionSum_i[:,1] = localSum_i
  #Bradcast total blockSum
  for id = 1 : (nProc-1)
    MPI.Recv!( localSumOther_r, id, 1, comm )
    MPI.Recv!( localSumOther_i, id, 2, comm )
    projectionSum_r[:,id+1] = localSumOther_r
    projectionSum_i[:,id+1] = localSumOther_i
  end
  projections_r = sum( projectionSum_r, 2 )
  projections_i = sum( projectionSum_i, 2 )
#   println( projections_r )
#   println( timeSamples )
else 
  MPI.Send( localSum_r, root, 1, comm )
  MPI.Send( localSum_i, root, 2, comm )
end



finalTime = int( nIterations*dt )

#Save final state
if timeDirection == "backward"
  finalFileName = "finalPsiInit_$(pId_z)_$(pId_y)_$(pId_x).h5"
else
  finalFileName = "finalPsi_$(pId_z)_$(pId_y)_$(pId_x)_$(timeDirection)_t$(finalTime)_n$(nPoints).h5"
end

finalFile = h5open( snapDir * finalFileName, "w")
saveState_64( 0, finalFile )

if pId == root
  analysisFileName = "analysis/$(timeDirection).h5"
  analysisFile = h5open( dataDir * analysisFileName, "w")

  println( "\n\n[ pID: $pId ] Saving final state: " * finalFileName   )
  println( "[ pID: $pId ] Saving analysis file: " * analysisFileName   )
  
  analysisFile["projections_r"] = projections_r 
  analysisFile["projections_i"] = projections_i
  analysisFile["times"] = timeSamples
  close( analysisFile )
end
##################################################################
MPI.Barrier(comm)
if pId == root
  time = totalTime[1] + totalTime[2] + totalTime[3]
  print( "\n[ pID: $(pId) ] totalTime: $(time) secs\n" )
  print( "[ pID: $(pId) ] Compute time: $(totalTime[1]) secs,  ( $(totalTime[1]/time) )\n" )
  print( "[ pID: $(pId) ] Transfer time: $(totalTime[2]) secs,  ( $(totalTime[2]/time) )\n" )
  print( "[ pID: $(pId) ] Save time: $(totalTime[3]) secs,  ( $(totalTime[3]/time) )\n\n" )
end

close( snapshotsFile )
close( finalFile )
MPI.Barrier(comm)
MPI.Finalize()