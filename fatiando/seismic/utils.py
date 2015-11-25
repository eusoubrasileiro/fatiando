r"""
Tools for seismic modeling or inversion

**Auxiliary functions**

* :func:`~fatiando.seismic.utils.nmo2`: apply 2D nmo correction on CMP gather

* :func:`~fatiando.seismic.utils.`:

**Theory**

Nmo correction based on a simple horizontal layered earth is given by:

.. math::

    t^2 = (t_0)^2 + x^2/(v_rms)^2



"""

import numpy as np
from scipy import interpolate


def nmo(cmp_gather, offsets, vnmo, dt, stretching=0.4):
    r"""
    Given nmo functions defined by t0 and vrms, apply
    the nmo correction on the 2D cmp_gather of traces.
    t0 is the time sample

    Parameters:

    * cmp_gather : 2D-array
        traces of this gather from near to far-offset
        (nsamples, ntraces)
    * offsets : 1D-array
        off-sets for each cmp in this gather
    * vnmo : 1D-array
        velocity parameter of all nmo functions for this cmp gather
        must have same size as (nsamples)
    * dt : float
        sample rate
    * stretching: float
        percentage of frequency change due nmo stretching acceptable,
        above this limited, the trace region is muted
        uses delta nmo over t0 as criteria

    Returns:

    * cmp_nmo : 2D array
        nmo corrected cmp traces

    Note:

    Uses linear interpolation of sample values.

    """

    #  create output cmp_gather
    ns, nx = cmp_gather.shape
    cmp_nmo = np.zeros(cmp_gather.shape)

    for j in range(nx):  # correct each offset
        x = offsets[j]
        # function to interpolate amplitude values for this trace
        interpfunc = interpolate.interp1d(range(ns)*dt, cmp_gather[:, j])
        for i in range(ns):  # for each (t0, vrms) hyperbola of this trace
            t0 = i*dt
            t = np.sqrt(t0**2+(x/vnmo[i])**2)
            if stretching is not None:
                # dtnmo/t0 equivalent to df/f frequency distortion Oz. Yilmaz
                if (t-t0)/t0 > stretching:
                    # will remain zero
                    continue
            cmp_nmo[i, j] = interpfunc(t)

    return cmp_nmo


def vrms_n (n, vi, twt):
    """
    RMS velocity from layer 0 to layer n

    * n : int
        layer index to where vrms will be calculated
    * vi : ndarray
        interval velocity array
    * twt : ndarray
        two way time for each layer vi

    """
    vrms_n = 0.
    t0 = 0.
    for i in range(n):
        vrms_n += twt[i]*vi[i]**2
        t0 += twt[i]

    return np.sqrt(vrms_n/t0)


def vrms(vi, ds):
    """
    Calculate RMS velocity from:

    * vi : ndarray
        interval velocity array size
    * ds : ndarray
        layer size

    return Rms velocity array
    """
    twt = 2*(ds/vi) # two way time
    return [ vrms_n(i, vi, twt) for i in range(1,len(vi)+1) ]

