"""
Cython implementation of the time stepping functions.
"""
__all__ = ['_step_elastic_sh']

import numpy

from libc.math cimport exp
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T


def _step_elastic_sh(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] u_t,
    numpy.ndarray[DTYPE_T, ndim=2] u_tm1,
    int nx, int nz, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] svel,
    int pad, double decay):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    cdef int i, j
    cdef double in_pad
    for i in xrange(1, nz - 1):
        for j in xrange(1, nx - 1):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + (svel[i,j]**2)*(dt**2)*(
                    (u_t[i + 1,j] - 2.*u_t[i,j] + u_t[i - 1,j])/dz**2 +
                    (u_t[i,j + 1] - 2.*u_t[i,j] + u_t[i,j - 1])/dx**2))
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                u_tp1[i,j] *= exp(-in_pad**2/decay**2)
