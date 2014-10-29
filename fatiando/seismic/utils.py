r"""
Tools for seismic modeling or inversion

**Auxiliary functions**

* :func:`~fatiando.seismic.utils.nmo2`: apply 2D nmo correction on CMP gather

* :func:`~fatiando.seismic.utils.`:

**Theory**

Nmo correction based on a simple horizontal layered earth is given by:

.. math::

    t^2 = (t_0)^2 + x^2/(v_rms)^2

where Vrms is given by ... tchururu

"""

import numpy as np

def nmo2(t0, vrms, cmp_gather, dt, ds, stretching=0.4):
    r"""
    Given nmo functions defined by t0 and vrms, apply
    the nmo correction on the 2D cmp_gather of traces.

    Parameters:

    * t0 : 1D-array
        t0 parameter of all nmo functions for this cmp gather
    * vrms : 1D-array
        vrms paramater of all nmo functions for this cmp gather
    * cmp_gather : 2D-array
        traces of this gather from near to far-offset
    * dt : float
        sample rate
    * ds : float
        cmp space increment between off-sets
    * stretching: float
        percentage of frequency change due nmo stretching acceptable,
        above this limited, the trace region is muted
        Uses windowed fft to comparison.

    Returns:

    * cmp_nmo : 2D array
        nmo corrected cmp traces

    Note:

    Uses spline interpolation of sample values.

    """

    # create spline interpolations for all traces

    # create output cmp_gather
    cmp_nmo = np.zeros(cmp_gather.shape)

    for t, v in zip(t0, vrms):
        for i in range(cmp_gather.shape[1]):
            #for
            #cmp_nmo
            raise NotImplementedError(
            "We are working on it...")



    # perform nmo stretching check and mute

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

twt = 2*(ds/vi) # two way time