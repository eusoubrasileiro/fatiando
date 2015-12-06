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
from scipy.special import hankel2, jn, hankel1
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
    ws = numpy.pi*2*numpy.arange(0, (1./dt), (1./dt)/n)
    # all k's = w/c in omega/frequency domain to evaluate the solution
    ks = ws/c
    # hankel filter kernel
    hankel = -1j*numpy.pi*hankel2(0, ks*rho)
    hankel[0] = 1j  # i infinity in limit
    sw = numpy.fft.fft(source)  # just go to frequency
    return numpy.real(numpy.fft.ifft(hankel*sw))


def wedge_cylindrical(rho, phi, rho_s, phi_s, c, source, dt):
    """
    Analytic solution equation (3) paper Alford et. al. cylindrical coordinates
    for \rho (observation point) smaller equal \rho_s (source point)
    (look figure 1. in paper)
    90 degrees wedge model cylindrical coordinates
    
    R.M. Alford - Accuracy of Finite-Difference Modeling

    rho - radius from center position of observation point
    phi - angle in radians position of observation point
    rho_s - radius from center position of source function
    phi_s - angle in radians position of source function
    c - velocity    
    source - source fuction sampled in time
    dt - sample rate from source function in time

    """
    N = len(source)
    dw = numpy.pi*2*(1./dt)/N # omega increment for each omega that will be sampled 
    ks = numpy.array([(p*dw/c) for p in xrange(N)]) # all k's = w/c in omega/frequency domain to evaluate the solution
    
    # serie aproximation just 100 first terms
    serie = numpy.zeros(N) +1j * numpy.zeros(N)
    for n in xrange(1,100):
        tmp = jn(2*n/3, ks*rho)*hankel2(2*n/3,ks*rho_s)*numpy.sin(2*n*phi_s/3)*numpy.sin(2*n*phi/3)
        tmp[0] = 0. # bessel and hankel undefined in 0
        serie += tmp    
    
    sourcew = numpy.fft.fft(source) # source in the frequency domain    
    return numpy.real(numpy.fft.ifft(numpy.complex(0, -8*numpy.pi/3) * sourcew * serie))

def wedge_cylindrical_B(rho, phi, rho_s, phi_s, c, dt, m, alpha=1000):
    """
    Analytic solution equation (3) paper Alford et. al. cylindrical coordinates
    for \rho (observation point) smaller equal \rho_s (source point)
    (look figure 1. in paper)
    90 degrees wedge model cylindrical coordinates
    
    R.M. Alford - Accuracy of Finite-Difference Modeling

    rho - radius from center position of observation point
    phi - angle in radians position of observation point
    rho_s - radius from center position of source function
    phi_s - angle in radians position of source function
    c - velocity    
    dt - sample rate from source function in time
    m - number of samples
    alpha - source parameter equal 2f^2 of GaussSource
    use GaussSource f = sqrt(alpha/2)
    
    Experiment tha doesn't work.

    """
    N = m
    dw = numpy.pi*2*(1./dt)/N # omega increment for each omega that will be sampled 
    ks = numpy.array([(p*dw/c) for p in xrange(N)]) # all k's = w/c in omega/frequency domain to evaluate the solution
    # serie aproximation just 100 first terms
    serie = numpy.zeros(N) +1j * numpy.zeros(N)
    ks = -ks
    for n in xrange(1,100):
        tmp = jn(2*n/3, ks*rho)*hankel1(2*n/3,ks*rho_s)*numpy.sin(2*n*phi_s/3)*numpy.sin(2*n*phi/3)
        tmp[0] = 0. # bessel and hankel undefined in 0
        serie += tmp    
    # source in the frequency domain
    ws = ks/c # get omegas again
    sourcew = -(1j*ws/alpha)*numpy.sqrt(0.25*numpy.pi/alpha)*numpy.e**(-(0.25/alpha)*ws**2)
    return numpy.real(numpy.fft.ifft( (-1j*8*numpy.pi/3) * sourcew * serie))


def free_1d(x, c, source, dt):
    """
    Analytic solution equation 1D, using green function for
    helmoltz wave equation and fourier transform.
    1D free space.
    
    Doesn't work. A mistery for me!
    
    source at zero

    x - distance to source
    c - velocity
    source - source fuction sampled in time
    dt - sample rate from source function in time

    """
    N = len(source)
    # frequency increment for each frequency that will be sampled
    dw = numpy.pi*2*(1./dt)/N  
     # all k's = f/c in frequency domain to evaluate the solution 
    ks = numpy.array([(p*dw/c) for p in xrange(N)])   
    green = 0.5j*(numpy.cos(ks*x)+1j*numpy.sin(ks*x))/ks
    green[0] = 0.0
    sourcew = numpy.fft.fft(source) # source in the frequency domain
    return numpy.real(numpy.fft.ifft(green*sourcew))


def free_2d(rho, c, source, dt):
    """
    Analytic solution equation Alford et. al. cylindrical coordinates
    2D free space
    
    source at zero
    
    note: dot not evaluate at zero, solution not viable
    
    rho - distance to the source function
    c - velocity
    source - source fuction sampled in time
    dt - sample rate from source function in time

    """
    N = len(source)
    dw = numpy.pi*2*(1./dt)/N # omega increment for each omega that will be sampled
    # all k's = w/c in omega/frequency domain to evaluate the solution 
    ks = numpy.array([(p*dw/c) for p in xrange(N)])
    #hankelshift = -(1j*numpy.pi)*hankel2(0,ks*rho)
    # if I change the signal in the hankel function I can use the first kind
    hankelshift = -(1j*numpy.pi)*hankel2(0,ks*rho)
    hankelshift[0] = 0. # i infinity in limit
    
    sourcew = numpy.fft.fft(source) # source in the frequency domain  
    return numpy.real(numpy.fft.ifft(hankelshift*sourcew)) 


def _2cylindrical(x, z, x0=0, z0=0):
    """
    Convert from cartezian coordinates to cylindrical coordinates
    x, z increses right and downward. The origin is (x0, z0)
    where x,z represents the R^2 such (x, z, y) = (i, j, k)
    
    !include better description of the two coordinate systems.!
    
    Returns:
    
    rho - radius
    phi - angle
    
    """
    # having z increasing upward would remove the -
    return ( numpy.sqrt((x-x0)**2+(z-z0)**2), (2*numpy.pi-numpy.arctan2(z-z0,x-x0))%(2*numpy.pi) )

def _2cartezian(rh, phi, rh0, phi0):
    pass





