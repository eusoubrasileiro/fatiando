"""
Seismic: 3D finite difference simulation of Equivalent Staggered Grid (ESG)
acoustic wave equation scheme of Di Bartolo et al. (2012).
North and Down velocity gradient
"""
import numpy as np
from fatiando.seismic import wavefd

# Set the parameters of the finite difference grid 3D
shape = (100, 100, 100)
ds = 10.  # spacing
area = [0, shape[0]*ds, 0, shape[1]*ds, 0, shape[2]*ds]
# Set the parameters of the finite difference grid
velocity = np.ones(shape)*1500.  # m/s
density = np.ones(shape)*1000.  # kg/m^3
for i in xrange(100):  # density/velocity changing
    velocity[:, :, i] += i*20  # m/s
    # increasing with depth
    velocity[i, :, :] += i*20  # m/s
# avoiding spatial alias, frequency of source should be smaller than this
fc = 0.5*np.min(velocity)/ds  # based on plane waves v=l*f
fc -= 0.5*fc
sources = [wavefd.GaussSource((50*ds, 50*ds, 40*ds), area, shape,  10**(-8), fc)]
dt = wavefd.maxdt(area, shape, np.max(velocity))
duration = 0.35
maxit = int(duration/dt)
# x, y, z coordinate of the seismometer
stations = [[45*ds, 45*ds, 65*ds], [65*ds, 65*ds, 30*ds]]
snapshots = 5  # every 1 iterations plots one
simulation = wavefd.acoustic3_esg(velocity, density, area, dt, maxit,
                                sources, stations, snapshots)

from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction
from tvtk.util.ctf import PiecewiseFunction
import sys
# doesn't  work with ui=False
@mlab.animate(delay=100)
def anim():
    fig = mlab.figure(size=(600,600))
    t, u, seismogram = simulation.next()
    u = u.transpose()[::-1]
    sscalar = mlab.pipeline.scalar_field(u)
    extent = [0, shape[0], 0, shape[1], 0, shape[2]]
    mlab.axes(extent=extent)
    mlab.outline(extent=extent)
    #azimuth = 27.087178769208965
    #elevation = -120.9368000828039
    #distance = 334.60652149512919
    #focalpoint = [53.01525703,  57.20435378,  61.16758842]
    #mlab.view(azimuth, elevation, distance, focalpoint, reset_roll=True)
    svolume = mlab.pipeline.volume(sscalar)
    svolume.volume_mapper.lock_sample_distance_to_input_spacing = True
    svolume.volume_mapper.blend_mode = 'maximum_intensity'
    #svolume.volume_property.shade = 0 # remove shade better visualization
    #sv.voloume.property.get_scalar_opacity    
    for t, u, seismogram in simulation:
        minu = u.min(); maxu = u.max()
        # compress the color bar? for what reason?
        #minu = min+0.65*(maxuu-minuu); maxu = minuu+0.9*(maxuu-minuu)
        #min = min+0.65*(max-min); max = min+0.9*(max-min)
        # this here is not the wisest way (use lookup table for custom colorbar)
        # or use color for standard ones
        # changing colorbar! first (RedWhiteBlue)
        # Changing the otf: (opacity correspondence) also zero centric
        #ctf = ColorTransferFunction()
        ##ctf.add_rgb_point(value, r, g, b)  # r, g, and b are float between 0 and 1
        #ctf.add_rgb_point(minu, 1., 0., 0.)
        #ctf.add_rgb_point(0., 0., 0., 0.)
        #ctf.add_rgb_point(maxu, 0., 0., 1.)
        #svolume._volume_property.set_color(ctf)
        #svolume._ctf = ctf
        #svolume.update_ctf = True
        #otf = PiecewiseFunction()
        #otf.add_point(minu, 0.4)
        #otf.add_point(0., 0.0)
        #otf.add_point(maxu, 0.4)
        #svolume._otf = otf
        #svolume._volume_property.set_scalar_opacity(otf)
        u = u.transpose()[::-1]  # solving z up
        svolume.mlab_source.scalars = u
        sys.stdout.write("\rprogressing .. %.1f%% time %.3f"%(100.0*float(t)/maxit, (dt*t)))
        sys.stdout.flush()
        yield
        # the data created on the simulation is lost...
        # since the iterator is just a loop needed... nothing more...

anim()
mlab.show()  # make possible to manipulate at the end

#vtk manuals
#http://www.dcs.ed.ac.uk/teaching/cs4/www/visualisation/vtk/vtkhtml/manhtml/vtkVolumeProperty.html#toc5
#void SetColor( vtkColorTransferFunction *function );

# lessons so far. add opacity function (considering zero centric pattern on simulation)
# and try to implement a custom color bar with LUT