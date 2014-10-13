"""
Seismic: 3D finite difference
North and Down velocity gradient
"""
import numpy as np
import sys
from fatiando.seismic import wavefd

# Set the parameters of the finite difference grid 3D
shape = (100, 100, 100)
ds = 10.  # spacing
area = [0, shape[0]*ds, 0, shape[1]*ds, 0, shape[2]*ds]
# Set the parameters of the finite difference grid
velocity = np.ones(shape)*2500.  # m/s
velocity[50:100, 50:100, 50:100] = 1500. # m/s
# avoiding spatial alias, frequency of source should be smaller than this
fc = 0.5*np.min(velocity)/ds  # based on plane waves v=l*f
fc -= 0.5*fc
sources = [wavefd.GaussSource((40*ds, 40*ds, 30*ds), area, shape,  10**(-8), fc)]
dt = wavefd.maxdt(area, shape, np.max(velocity))
duration = 0.6
maxit = int(duration/dt)
# x, y, z coordinate of the seismometer
stations = [[45*ds, 45*ds, 65*ds], [65*ds, 65*ds, 30*ds]]
snapshots = 5  # every 1 iterations plots one
simulation = wavefd.scalar3(velocity, area, dt, maxit,
                                sources, stations, snapshots)
movie = np.zeros(((maxit/snapshots)+2, 100, 100, 100))
i = 0
for t, u, seismogram in simulation:
    movie[i] = u
    sys.stdout.write("\rprogressing .. %.1f%% time %.3f"%(100.0*float(t)/maxit, (dt*t)))
    sys.stdout.flush()
    i += 1