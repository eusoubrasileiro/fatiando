r"""
Tools for seismic inversion: migration

**Auxiliary functions**

* :func:`~fatiando.seismic.zft_rtm`: apply 2D reverse time depth migration in a Zero-oFfseT section

* :func:`~fatiando.seismic.shot_rtm`: apply 2D reverse time depth migration in a shot gather

Those implementations use explicit finite differences time and space for forward and reverse
time extrapolations of the wave field.

**Theory**

"""

import sys
import numpy
from scipy import signal
from fatiando.seismic import wavefd

def rt_scalar(vel, area, dt, iterations, boundary, snapshot=None, padding=-1, taper=0.006):
    """

    Simulate reverse in time scalar waves using an explicit finite differences scheme 4th order
    space. Uses a boundary condition at z=0, re-inserting the recorded values back on the
    wave-field simulation from the last values to the first.
    Used to make reverse time depth migration of zero-offset sections or shot gathers.

    The top implements a free-surface boundary condition (TODO: change to absorbing).
    For the left, right and lower uses boundaries uses Transparent condition of Reynolds, A. C.
    (Boundary conditions for numerical solution of wave propagation problems Geophysics p 1099-1110 - 1978)

    Parameters:

    * vel : 2D-array (defines shape simulation)
        The wave velocity at all the grid nodes, must be half the original velocity.
        The depth velocity model.
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
        at zmin, the deepest at zmax.
    * dt : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * boundary : 2D-array
        Those are the boundary values at z=0 for all iteration times.
        For zero-offset section migration, shot-gather migration
        this is a matrix of traces.
        Boundary must have same shape as vel and sample rate must be equal of dt.
    * snapshot : None or int
        If not None, than yield a snapshot of the scalar quantity disturbed at every
        *snapshot* iterations.
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
        default 5 percent nz
    * taper : float  (TODO: implement real gaussian)
        The intensity of the Gaussian taper function used for the absorbing
        boundary conditions. Adjust it for better absorption.

    Yields:

    * i, u : int, 2D-array
        The current iteration, the scalar quantity disturbed

    The last iteration is the migrated section in depth

    """

    if boundary.shape[1] != vel.shape[1]:  # just x must be equal
        raise IndexError("boundary must have same shape as velocity")
    if iterations != boundary.shape[0]:
        raise IndexError("Same number of interations needed for rtm")

    nz, nx = numpy.shape(vel) # get simulation dimensions
    x1, x2, z1, z2 = area
    dz, dx = (z2 - z1)/(nz - 1), (x2 - x1)/(nx - 1)

    # Add some padding to x and z. The padding region is where the wave is
    # absorbed by gaussian dumping
    pad = int(padding)
    if pad == -1:   # default 5% percent nz
        pad = int(0.05*nz) + 2  # plus 2 due 4th order
    nx += 2*pad
    nz += pad
    # Pad the velocity as well
    vel_pad = wavefd._add_pad(vel, pad, (nz, nx))
    # Pack the particle position u at 3 different times in one 3d array
    u = numpy.zeros((3, nz, nx), dtype=numpy.float)
    # insert the zero-offset samples reversed in time last ones first. For utp1 at z=0 for every x
    for j in xrange(nx-2*pad):  # tp1
        u[0, 0, j + pad] = boundary[iterations-1, j]
    if snapshot is not None:
        yield 0, u[0, :-pad, pad:-pad]
    for j in xrange(nx-2*pad):  # t
        u[1, 0, j + pad] = boundary[iterations-2, j]
    if snapshot is not None:
        yield 1, u[1, :-pad, pad:-pad]
    for iteration in xrange(2, iterations):
        tm1, t, tp1 = iteration % 3, (iteration-1) % 3, (iteration-2) % 3  # to avoid copying between panels
        # invert the order of the input parameters to make it reverse in time
        wavefd._step_scalar(u[tm1], u[t], u[tp1], 2, nx - 2, 2, nz - 2,
                     dt, dx, dz, vel_pad)
        # _apply_damping(u[tp1], nx-2, nz-2, pad-2, taper)
        # forth order +2-2 indexes needed
        # apply Reynolds 1d plane wave absorbing condition
        wavefd._nonreflexive_scalar_boundary_conditions(u[tm1], u[t], u[tp1], vel_pad, dt, dx, dz, nx, nz)
        # insert the zero-offset samples reversed in time last ones first. For utp1 at z=0 for every x
        for j in xrange(nx-2*pad):
            u[t, 0, j + pad] = boundary[iterations-(iteration+1), j]
        if snapshot is not None and iteration%snapshot == 0:
            yield iteration, u[tm1, :-pad, pad:-pad]


def pre_rtmshot(shot, dt, vdepth, area, fc, source):
    """
    Perform pre-stack reverse in time depth migration on a 2D shot gather,
    Forward and reverse modelling of shots are done using scalar wave equation.
    For image condition uses the normalized cross-correlation of every grid node.

    Parameters:

    * shot : 2D-array
        The shot gather, time x space
    * dt : float
        sample rate
    * vdepth : 2D-array
        The depth velocity field at all receiver positions
    * area : [xmin, xmax, zmin, zmax]
         The x, z limits of the shot/velocity area, e.g., the shallowest point is
         at zmin, the deepest at zmax
    * fc : source frequency
        Used for forward modelling based on a Gauss Source
    * source: (sx, sz)
        x, z coordinates of source source

     Returns:

    * migrated shot : 2D
        the depth migrated shot same shape as vdepth

    """
    # Basic parameters
    # Set the parameters of the finite difference grid
    nz, nx = vdepth.shape
    x1, x2, z1, z2 = area
    dz, dx = (z2 - z1) / (nz - 1), (x2 - x1) / (nx - 1)
    ns = shot.shape[0]  # number samples per trace

    # avoiding spatial alias and numerical dispersion based on plane waves v=l*f and Alford et al.
    # # and using at least 5 points per wavelength
    eps = 0.98*1./(5*max(dx, dz)*min(1./(2*dx), 1./(2*dz)))
    idealfc = eps*numpy.min(vdepth)/(max(2*dx, 2*dz))
    if fc > idealfc:
        sys.stdout.write("Warning: the simulation might have strong numerical dispersion making it unusable\n")
        sys.stdout.write("Warning: consider using a finer velocity model")

    simsource = [wavefd.GaussSource(source, area, (nz, nx),  1., fc)]  # forward simulation source
    simdt = wavefd.scalar_maxdt(area, vdepth.shape, numpy.max(vdepth))  # forward simulation time step
    simit = int(numpy.floor(ns*dt/simdt))  # maximum number of iterations needed for forward modelling
    # run forward modelling of the shot
    fwdsimulation = wavefd.scalar(vdepth, area, simdt, simit, simsource, snapshot=1, padding=50)

    # dt from signal must be equal to dt from simulation, so resample it first
    # resample the input signal is better then resampling everything else
    simshot = shot
    if dt != simdt:  #  resample shot if needed
        if dt > simdt:  # low pass filtering on Nyquest first of shot sample rate
            # 1/(2*simdt) is equal of Nyquest=1 for the input signal
            b, a = signal.butter(8, dt/simdt)
            simshot = signal.filtfilt(b, a, shot, axis=0)
        simshot = signal.resample(simshot, simit, axis=0)

    # run the forward simulation and record every time step of the grid
    fwdfield = numpy.zeros((simit, nz, nx))
    for i, u, seismograms in fwdsimulation:
        fwdfield[i, :, :] = u
        sys.stdout.write("\rforward modeling progressing .. %.1f%% time %.3f" % (100.0*float(i)/simit, (simdt*i)))
        sys.stdout.flush()

    # Reverse in time shot basic parameters same from forward modelling
    rtmsimulation = rt_scalar(vdepth, area, simdt, simit, simshot, snapshot=1, padding=50)

    # run the reverse time simulation and record every time step of the grid
    rtmfield = numpy.zeros((simit, nz, nx))
    for i, u in rtmsimulation:
        rtmfield[i, :, :] = u
        sys.stdout.write("\rreverse in time modeling progressing .. %.1f%% time %.3f" % (100.0*float(i)/simit, (simdt*i)))
        sys.stdout.flush()

    # normalized cross-correlation image condition
    migratedshot = numpy.zeros((nz, nx))
    for i in xrange(nz):
        for j in xrange(nx):
            migratedshot[i, j] = numpy.dot(rtmfield[:, i, j], fwdfield[::-1, i, j])
            migratedshot[i, j] /= numpy.sum(fwdfield[:, i, j]**2)

    return migratedshot

# def rt_scalar(vel, area, dt, iterations, boundary, snapshot=None, padding=-1, taper=0.006):
#     """
#
#     Simulate reverse in time scalar waves using an explicit finite differences scheme 4th order
#     space. Uses a boundary condition at z=0, re-inserting the recorded values back on the
#     wave-field simulation from the last values to the first.
#     Used to make reverse time depth migration of zero-offset sections or shot gathers.
#
#     The top implements a free-surface boundary condition (TODO: change to absorbing).
#     For the left, right and lower uses boundaries uses Transparent condition of Reynolds, A. C.
#     (Boundary conditions for numerical solution of wave propagation problems Geophysics p 1099-1110 - 1978)
#
#     Parameters:
#
#     * vel : 2D-array (defines shape simulation)
#         The wave velocity at all the grid nodes, must be half the original velocity.
#         The depth velocity model.
#     * area : [xmin, xmax, zmin, zmax]
#         The x, z limits of the simulation area, e.g., the shallowest point is
#         at zmin, the deepest at zmax.
#     * dt : float
#         The time interval between iterations
#     * iterations : int
#         Number of time steps to take
#     * boundary : 2D-array
#         Those are the boundary values at z=0 for all iteration times.
#         For zero-offset section migration, shot-gather migration
#         this is a matrix of traces.
#         Boundary must have same shape as vel and sample rate must be equal of dt.
#     * snapshot : None or int
#         If not None, than yield a snapshot of the scalar quantity disturbed at every
#         *snapshot* iterations.
#     * padding : int
#         Number of grid nodes to use for the absorbing boundary region
#         default 5 percent nz
#     * taper : float  (TODO: implement real gaussian)
#         The intensity of the Gaussian taper function used for the absorbing
#         boundary conditions. Adjust it for better absorption.
#
#     Yields:
#
#     * i, u : int, 2D-array
#         The current iteration, the scalar quantity disturbed
#
#     The last iteration is the migrated section in depth
#
#     """
#
#     if boundary.shape[1] != vel.shape[1]:  # just x must be equal
#         raise IndexError("boundary must have same shape as velocity")
#     if iterations != boundary.shape[0]:
#         raise IndexError("Same number of interations needed for rtm")
#
#     nz, nx = numpy.shape(vel) # get simulation dimensions
#     x1, x2, z1, z2 = area
#     dz, dx = (z2 - z1)/(nz - 1), (x2 - x1)/(nx - 1)
#
#     # Add some padding to x and z. The padding region is where the wave is
#     # absorbed by gaussian dumping
#     pad = int(padding)
#     if pad == -1:   # default 5% percent nz
#         pad = int(0.05*nz) + 2  # plus 2 due 4th order
#     nx += 2*pad
#     nz += pad
#     # Pad the velocity as well
#     vel_pad = wavefd._add_pad(vel, pad, (nz, nx))
#     # Pack the particle position u at 3 different times in one 3d array
#     u = numpy.zeros((3, nz, nx), dtype=numpy.float)
#     # insert the zero-offset samples reversed in time last ones first. For utp1 at z=0 for every x
#     for j in xrange(nx-2*pad):  # tp1
#         u[0, 0, j + pad] = boundary[iterations-1, j]
#     if snapshot is not None:
#         yield 0, u[0, :-pad, pad:-pad]
#     for j in xrange(nx-2*pad):  # t
#         u[1, 0, j + pad] = boundary[iterations-2, j]
#     if snapshot is not None:
#         yield 1, u[1, :-pad, pad:-pad]
#     for iteration in xrange(2, iterations):
#         tm1, t, tp1 = iteration % 3, (iteration-1) % 3, (iteration-2) % 3  # to avoid copying between panels
#         # invert the order of the input parameters to make it reverse in time
#         wavefd._step_scalar(u[tm1], u[t], u[tp1], 2, nx - 2, 2, nz - 2,
#                      dt, dx, dz, vel_pad)
#         # _apply_damping(u[tp1], nx-2, nz-2, pad-2, taper)
#         # forth order +2-2 indexes needed
#         # apply Reynolds 1d plane wave absorbing condition
#         wavefd._nonreflexive_scalar_boundary_conditions(u[tm1], u[t], u[tp1], vel_pad, dt, dx, dz, nx, nz)
#         # insert the zero-offset samples reversed in time last ones first. For utp1 at z=0 for every x
#         for j in xrange(nx-2*pad):
#             u[t, 0, j + pad] = boundary[iterations-(iteration+1), j]
#         if snapshot is not None and iteration%snapshot == 0:
#             yield iteration, u[tm1, :-pad, pad:-pad]
#     yield iteration, u[tm1, :-pad, pad:-pad]