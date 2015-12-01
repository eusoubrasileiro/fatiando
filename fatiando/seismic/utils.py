r"""
Tools for seismic processing

**Auxiliary functions**

* :func:`~fatiando.seismic.utils.vrms`: Calculate RMS velocity from interval
thickness and velocity (horizontally layered model)

* :func:`~fatiando.seismic.utils.nmo`: Apply nmo correction on a CMP gather

* :func:`~fatiando.seismic.utils.plot_vnmo`: Draw nmo hyperbolas over a CMP gather

**Theory**

Nmo correction based on a simple horizontal layered earth is given by:

.. math::

    t^2 = (t_0)^2 + x^2/(v_rms)^2

"""

import numpy as np
from scipy import interpolate
from fatiando.vis import mpl


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
        in the same sequence as ntraces
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
        interpfunc = interpolate.interp1d(
            np.linspace(0, ns*dt, ns), cmp_gather[:, j])
        for i in range(ns):  # for each (t0, vnmo) hyperbola of this trace
            t0 = i*dt
            t = np.sqrt(t0**2+(x/vnmo[i])**2)
            if t > ns*dt:  # maximum time to correct
                continue
            if stretching is not None:
                # dtnmo/t0 equivalent to df/f frequency distortion Oz. Yilmaz
                if (t-t0)/t0 > stretching:
                    # will remain zero
                    continue
            cmp_nmo[i, j] = interpfunc(t)

    return cmp_nmo


def vrms(vi, ds):
    """
    Calculate RMS (Root Mean Square) velocity from interval thickness
    and velocity (horizontally layered model)

    * vi : ndarray
        interval velocity
    * ds : ndarray
        layer size

    return Rms velocity array
    """
    twt = 2*(ds/vi)  # two way time
    vi2_twt = (vi**2)*twt
    return np.sqrt(vi2_twt.cumsum()/twt.cumsum())


def plot_vnmo(cmp_gather, offsets, vnmo, dt, inc=70,
              vmin=None, vmax=None, aspect='auto'):
    r"""
    Given nmo functions defined by t0 and vnmo, draw
    it over the specified gather using `seismic_image`

    Parameters:

    * cmp_gather : 2D-array
        traces of this gather from near to far-offset
        (nsamples, ntraces)
    * offsets : 1D-array
        off-sets for each cmp in this gather
        in the same sequence as ntraces
    * vnmo : 1D-array
        velocity parameter of all nmo functions for this cmp gather
        must have same size as (nsamples)
    * dt : float
        sample rate
    * inc: int
        plotting option, step in time samples between hyperbolas
        to avoid overlapping totally the seismic image bellow
    * vmin, vmax : float
        min and max values for imshow
    * aspect : float
        matplotlib imshow aspect parameter, ratio between axes


    Returns:

    * cmp_nmo : 2D array
        nmo corrected cmp traces

    """
    ns, nx = cmp_gather.shape

    for i in range(0, ns, inc): # each (t0, vnmo) hyperbola
        t0 = i*dt
        t = np.sqrt(t0**2+(offsets/vnmo[i])**2)
        mpl.seismic_image(cmp_gather, dt, aspect=aspect, vmin=vmin, vmax=vmax)
        mpl.plot(range(nx), t, '+b')