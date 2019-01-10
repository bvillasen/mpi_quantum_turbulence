import yt
import numpy as np
import matplotlib.pyplot as plt

nPoints = 128
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
Lx = 5.0
Ly = 5.0
Lz = 5.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]
arr = np.cos(X)*np.cos(Y)*np.cos(Z) + 20

data = { 'density': (arr, "g/cm**3") }
bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
ds = yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=64)

#slc = yt.SlicePlot(ds, "z", ["density"])
#slc.set_cmap("density", "Blues")
#slc.annotate_grids(cmap=None)
#slc.show()

#Find the min and max of the field
mi, ma = ds.all_data().quantities.extrema('density')
##Reduce the dynamic range
#mi = mi.value + 1.5e7
#ma = ma.value - 0.81e7

tf = yt.ColorTransferFunction((mi, ma), )

# Choose a vector representing the viewing direction.
L = [0.5, 0.5, 0.5]
# Define the center of the camera to be the domain center
c = ds.domain_center[0]
# Define the width of the image
W = 1.5*ds.domain_width[0]
# Define the number of pixels to render
Npixels = 512 

cam = ds.camera(c, L, W, Npixels, tf, fields=['density'],
                north_vector=[0,0,1], steady_north=True, 
                sub_samples=5, log_fields=[False])

tf.add_layers(5, 0.01, colormap = 'jet')
im = cam.snapshot('test_rendering.png')

#cam.transfer_function.map_to_colormap(mi,ma, 
                                      #scale=15.0, colormap='algae')
#cam.save_image('tes.png')

#cam.show()