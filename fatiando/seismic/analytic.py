"""
Analytic solution wave equation 2D for free space.

Using cylindrical coordinates for a free a space medium homogeneous of
velocity **$v$**. If the observer is at a **$\rho$** distance from the source.
The analytical solution is written from the frequency domain solution as:

math::

    U(\rho, t) = \frac{1}{2\pi}\int_{-\inf}^{+\inf}-F(\omega)
    i\pi H^{(2)}_{0}(k\rho)


where $F(\omega)$ is the fourier transform of the source
function $f(t)$

$H^{(2)}$ is the second Hankel function of order 0

"""
from scipy.special import hankel2
from scipy import signal
import numpy


def wave2d_analytic(rho, c, dt, source):
    """
    Analytic solution wave equation 2D for free space.
    Assume that source is at rho=0

    Do not evaluate at zero, solution not real!

    * rho : float
        distance to the source function
    * c : float
        velocity
    * dt : float
        sample rate for time source and analytical solution
    * source : 1D array
        source function in time

    References:

    Alford, R. M., Kelly, K. R., Boore, D. M. (1974) Accuracy of
    finite-difference modeling of the acoustic wave equation.
    Geophysics, 39(06), 834-842

    """
    n = source.size
    if rho/c > dt*n:  # make enough time for final solution
        newn = (rho/c)/dt + 16
        source = signal.resample(source, newn)
        n = newn
    # omega (2pi*f) increment for each omega that will be sampled
    ws = numpy.pi*2*numpy.arange(0, (1./dt), (1./(dt*n)))
    # all k's = w/c in omega/frequency domain to evaluate the solution
    ks = ws/c
    print ws.size
    # hankel filter kernel
    hankel = -1j*numpy.pi*hankel2(0, ks*rho)
    hankel[0] = 1j  # i infinity in limit
    sw = numpy.fft.fft(source)  # just go to frequency
    return numpy.real(numpy.fft.ifft(hankel*sw))
