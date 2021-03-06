"""
Potential field transformations, like upward continuation and derivatives.

.. note:: Most, if not all, functions here required gridded data.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of
  gridded potential field data on a level surface.
* :func:`~fatiando.gravmag.transform.reduce_to_pole`: Reduce the total field
  magnetic anomaly to the pole.
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction (North-South)
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction (East-West)
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

----

"""
from __future__ import division
import numpy

from .. import utils


def reduce_to_pole(x, y, data, shape, inc, dec, sinc, sdec):
    r"""
    Reduce total field magnetic anomaly data to the pole.

    The reduction to the pole if a phase transformation that can be applied to
    total field magnetic anomaly data. It "simulates" how the data would be if
    **both** the Geomagnetic field and the magnetization of the source were
    vertical (:math:`90^\circ` inclination) (Blakely, 1996).

    This functions performs the reduction in the frequency domain (using the
    FFT). The transform filter is (in the frequency domain):

    .. math::

        RTP(k_x, k_y) = \frac{|k|}{
            a_1 k_x^2 + a_2 k_y^2 + a_3 k_x k_y +
            i|k|(b_1 k_x + b_2 k_y)}

    in which :math:`k_x` and :math:`k_y` are the wave-numbers in the x and y
    directions and

    .. math::

        |k| = \sqrt{k_x^2 + k_y^2} \\
        a_1 = m_z f_z - m_x f_x \\
        a_2 = m_z f_z - m_y f_y \\
        a_3 = -m_y f_x - m_x f_y \\
        b_1 = m_x f_z + m_z f_x \\
        b_2 = m_y f_z + m_z f_y

    :math:`\mathbf{m} = (m_x, m_y, m_z)` is the unit-vector of the total
    magnetization of the source and
    :math:`\mathbf{f} = (f_x, f_y, f_z)` is the unit-vector of the Geomagnetic
    field.

    .. note:: Requires gridded data.

    .. warning::

        The magnetization direction of the anomaly source is crucial to the
        reduction-to-the-pole.
        **Wrong values of *sinc* and *sdec* will lead to a wrong reduction.**

    Parameters:

    * x, y : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field anomaly data at each point.
    * shape : tuple = (nx, ny)
        The shape of the data grid
    * inc, dec : floats
        The inclination and declination of the inducing Geomagnetic field
    * sinc, sdec : floats
        The inclination and declination of the total magnetization of the
        anomaly source. The total magnetization is the vector sum of the
        induced and remanent magnetization. If there is only induced
        magnetization, use the *inc* and *dec* of the Geomagnetic field.

    Returns:

    * rtp : 1d-array
        The data reduced to the pole.

    References:

    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    fx, fy, fz = utils.ang2vec(1, inc, dec)
    if sinc is None or sdec is None:
        mx, my, mz = fx, fy, fz
    else:
        mx, my, mz = utils.ang2vec(1, sinc, sdec)
    kx, ky = [k for k in _fftfreqs(x, y, shape, shape)]
    kz_sqr = kx**2 + ky**2
    a1 = mz*fz - mx*fx
    a2 = mz*fz - my*fy
    a3 = -my*fx - mx*fy
    b1 = mx*fz + mz*fx
    b2 = my*fz + mz*fy
    # The division gives a RuntimeWarning because of the zero frequency term.
    # This suppresses the warning.
    with numpy.errstate(divide='ignore', invalid='ignore'):
        rtp = (kz_sqr)/(a1*kx**2 + a2*ky**2 + a3*kx*ky +
                        1j*numpy.sqrt(kz_sqr)*(b1*kx + b2*ky))
    rtp[0, 0] = 0
    ft_pole = rtp*numpy.fft.fft2(numpy.reshape(data, shape))
    return numpy.real(numpy.fft.ifft2(ft_pole)).ravel()


def upcontinue(x, y, data, shape, height):
    r"""
    Upward continuation of potential field data.

    Calculates the continuation through the Fast Fourier Transform in the
    wavenumber domain (Blakely, 1996):

    .. math::

        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

    and then transformed back to the space domain. :math:`h_{up}` is the upward
    continue data, :math:`\Delta z` is the height increase, :math:`F` denotes
    the Fourier Transform,  and :math:`|k|` is the wavenumber modulus.

    .. note:: Requires gridded data.

    .. note:: x, y, z and height should be in meters.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * height : float
        The height increase (delta z) in meters.

    Returns:

    * cont : array
        The upward continued data

    References:

    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    assert x.shape == y.shape, \
        "x and y arrays must have same shape"
    assert height > 0, \
        "Continuation height increase 'height' should be positive"
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    kz = numpy.sqrt(kx**2 + ky**2)
    upcont_ft = numpy.fft.fft2(padded)*numpy.exp(-height*kz)
    cont = numpy.real(numpy.fft.ifft2(upcont_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont


def _upcontinue_space(x, y, data, shape, height):
    """
    Upward continuation using the space-domain formula.

    DEPRECATED. Use the better implementation using FFT. Kept here for
    historical reasons.

    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    dy = (y.max() - y.min())/(ny - 1)
    area = dx*dy
    deltaz_sqr = (height)**2
    cont = numpy.zeros_like(data)
    for i, j, g in zip(x, y, data):
        cont += g*area*((x - i)**2 + (y - j)**2 + deltaz_sqr)**(-1.5)
    cont *= abs(height)/(2*numpy.pi)
    return cont


def tga(x, y, data, shape, method='fd'):
    r"""
    Calculate the total gradient amplitude (TGA).

    This the same as the `3D analytic signal` of Roest et al. (1992), but we
    prefer the newer, more descriptive nomenclature suggested by Reid (2012).

    The TGA is defined as the amplitude of the gradient vector of a potential
    field :math:`T` (e.g. the magnetic total field anomaly):

    .. math::

        TGA = \sqrt{
            \left(\frac{\partial T}{\partial x}\right)^2 +
            \left(\frac{\partial T}{\partial y}\right)^2 +
            \left(\frac{\partial T}{\partial z}\right)^2 }

    .. note:: Requires gridded data.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units and so will the total gradient amplitude! I strongly
        recommend converting the data to SI **before** calculating the
        TGA is you need the gradient in Eotvos (use one of the unit conversion
        functions of :mod:`fatiando.utils`).

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * method : string
        The method used to calculate the horizontal derivatives. Options are:
        ``'fd'`` for finite-difference (more stable) or ``'fft'`` for the Fast
        Fourier Transform. The z derivative is always calculated by FFT.

    Returns:

    * tga : 1D-array
        The amplitude of the total gradient

    References:

    Reid, A. (2012), Forgotten truths, myths and sacred cows of Potential
    Fields Geophysics - II, in SEG Technical Program Expanded Abstracts 2012,
    pp. 1-3, Society of Exploration Geophysicists.

    Roest, W., J. Verhoef, and M. Pilkington (1992), Magnetic interpretation
    using the 3-D analytic signal, GEOPHYSICS, 57(1), 116-125,
    doi:10.1190/1.1443174.

    """
    dx = derivx(x, y, data, shape, method=method)
    dy = derivy(x, y, data, shape, method=method)
    dz = derivz(x, y, data, shape)
    res = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return res


def derivx(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the x direction.

    .. note:: Requires gridded data.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        kx, _ = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(kx*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dx = (x.max() - x.min())/(nx - 1)
        deriv = numpy.empty_like(datamat)
        deriv[1:-1, :] = (datamat[2:, :] - datamat[:-2, :])/(2*dx)
        deriv[0, :] = deriv[1, :]
        deriv[-1, :] = deriv[-2, :]
        if order > 1:
            deriv = derivx(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivy(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the y direction.

    .. note:: Requires gridded data.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        _, ky = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(ky*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dy = (y.max() - y.min())/(ny - 1)
        deriv = numpy.empty_like(datamat)
        deriv[:, 1:-1] = (datamat[:, 2:] - datamat[:, :-2])/(2*dy)
        deriv[:, 0] = deriv[:, 1]
        deriv[:, -1] = deriv[:, -2]
        if order > 1:
            deriv = derivy(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivz(x, y, data, shape, order=1, method='fft'):
    """
    Calculate the derivative of a potential field in the z direction.

    .. note:: Requires gridded data.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fft'`` for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    assert method == 'fft', \
        "Invalid method '{}'".format(method)
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    deriv_ft = numpy.fft.fft2(padded)*numpy.sqrt(kx**2 + ky**2)**order
    deriv = numpy.real(numpy.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx: padx + nx, pady: pady + ny].ravel()


def _pad_data(data, shape):
    n = _nextpow2(numpy.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = numpy.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
    return padded, padx, pady


def _nextpow2(i):
    buf = numpy.ceil(numpy.log(i)/numpy.log(2))
    return int(2**buf)


def _fftfreqs(x, y, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(padshape[1], dy)
    return numpy.meshgrid(fy, fx)[::-1]
