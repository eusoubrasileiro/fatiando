"""
Seismic: 2D finite difference simulation of scalar wave propagation.

Difraction example in cylindrical wedge model. Based on:
R. M. Alford, K. R. Kelly and D. M. Boore - 
Accuracy of finite-difference modeling of the acoustic wave equation. 
Geophysics  1974
"""
import numpy as np
from matplotlib import animation
from fatiando.seismic import wavefd
from fatiando.vis import mpl

# Set the parameters of the finite difference grid
shape = (200, 200)
dx = dz = 100. # spacing
area = [0, shape[1]*dx, 0, shape[0]*dz]
# Set the parameters of the finite difference grid
velocity = np.zeros(shape)+6000.
velocity[100:,100:] = 0.
fc = 15.
sources = [wavefd.GaussSource((125*dx, 75*dz), area, shape,  1., fc)]
dt = wavefd.scalar_maxdt(area, shape, np.max(velocity))
duration = 1.9
maxit = int(duration/dt)
stations = [[75*dx, 125*dz]] # x, z coordinate of the seismometer
snapshots = 3 # every 3 iterations plots one
simulation = wavefd.scalar(velocity, area, dt, maxit, sources, stations, snapshots)

# This part makes an animation using matplotlib animation API
background = (velocity-4000)*10**-1
fig = mpl.figure(figsize=(8, 6))
mpl.subplots_adjust(right=0.98, left=0.11, hspace=0.5, top=0.93)
mpl.subplot2grid((4, 3), (0,0), colspan=3,rowspan=3)
wavefield = mpl.imshow(np.zeros_like(velocity), extent=area, cmap=mpl.cm.gray_r,
                       vmin=-300, vmax=300)
mpl.points(stations, '^b', size=8)
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.m2km()
mpl.subplot2grid((4,3), (3,0), colspan=3)
seismogram1, = mpl.plot([],[],'-k')
mpl.xlim(0, duration)
mpl.ylim(-200, 200)
mpl.ylabel('Amplitude')
times = np.linspace(0, dt*maxit, maxit)
# This function updates the plot every few timesteps
def animate(i):
    t, u, seismogram = simulation.next()
    seismogram1.set_data(times[:t+1], seismogram[0][:t+1])
    wavefield.set_array(background[::-1]+u[::-1])
    return wavefield, seismogram1
anim = animation.FuncAnimation(fig, animate, frames=maxit/snapshots, interval=1)
mpl.show()