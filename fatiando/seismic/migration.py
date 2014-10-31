r"""
Tools for seismic inversion: migration

**Auxiliary functions**

* :func:`~fatiando.seismic.rtm`: apply 2D rtm seismic migration in a zero-offset section

**Theory**

"""

import numpy
from fatiando.seismic import wavefd

def rtmscalar(vel, area, dt, iterations, boundary, snapshot=None, padding=-1, taper=0.006):
    """

    Simulate reverse in time scalar waves using an explicit finite differences scheme 4th order
    space. For zero-offset reverse time migration.

    The top implements a free-surface boundary condition (TODO: change to absorbing).
    For the left, right and lower uses boundaries uses Transparent condition of Reynolds, A. C.
    (Boundary conditions for numerical solution of wave propagation problems Geophysics p 1099-1110 - 1978)

    Parameters:

    * vel : 2D-array (defines shape simulation)
        The wave velocity at all the grid nodes, must be half the original velocity for migration.
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
        at zmin, the deepest at zmax.
    * dt : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * boundary : 2D-array
        Zero-offset section of traces with same shape as vel
        sample rate must be equal of dt. Those are the boundary values at z=0.
    * stations : None or list
        If not None, then a list of [x, z] pairs with the x and z coordinates
        of the recording stations. These are physical coordinates, not the
        indexes
    * snapshot : None or int
        If not None, than yield a snapshot of the displacement at every
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

    """

    if boundary.shape != vel.shape:
        raise IndexError("boundary must have same shape as velocity")
    if iterations != boundary.shape[0]:
        raise IndexError("Same numer of interations needed for rtm")

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
    for j in xrange(nx):
        u[0, 0, j + pad] -= boundary[0][j]
    if snapshot is not None:
        yield 0, u[2, :-pad, pad:-pad]
    for iteration in xrange(1, iterations): # just invert the order to make it reverse in time
        tp1, t, tm1 = iteration % 3, (iteration+1) % 3, (iteration+2) % 3  # to avoid copying between panels
        # dumping not working with Reynolds need to fix apply dumping
        wavefd._step_scalar(u[tp1], u[t], u[tm1], 2, nx - 2, 2, nz - 2,
                     dt, dx, dz, vel_pad)
        # _apply_damping(u[tp1], nx-2, nz-2, pad-2, taper)
        # forth order +2-2 indexes needed
        # apply Reynolds 1d plane wave absorbing condition
        wavefd._nonreflexive_scalar_boundary_conditions(u[tp1], u[t], u[tm1], vel_pad, dt, dx, dz, nx, nz)
        # insert the zero-offset samples reversed in time last ones first. For utp1 at z=0 for every x
        for j in xrange(nx):
            u[tp1, 0, j + pad] -= boundary[iteration][j]
        if snapshot is not None and iteration%snapshot == 0:
            yield iteration, u[tp1, :-pad, pad:-pad]
    yield iteration, u[tp1, :-pad, pad:-pad]