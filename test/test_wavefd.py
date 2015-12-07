from __future__ import division
from fatiando.seismic.wavefd import (Ricker,
                                     Gauss,
                                     ElasticSH,
                                     ElasticPSV,
                                     Scalar)
from fatiando.seismic import analytic
import numpy as np
from numpy.testing import assert_almost_equal


def test_sources():
    "testing simple source parameters"
    w = Ricker(amp=10, cf=20)
    assert (w(0) == 10.0)  # maximum
    assert_almost_equal(w(-2), w(2), decimal=5,
                        err_msg="Ricker has no symmetry")
    w = Gauss(amp=10, cf=20)
    assert_almost_equal(w(0), 0.0, decimal=4,
                        err_msg="Gauss has no symmetry")
    assert_almost_equal(w(-2), w(2), decimal=5,
                        err_msg="Gauss has no symmetry")

def test_wavefd_elastipsv_run():
    "make a simple run of elastic psv"
    shape = (50, 50)
    pvel = 4000*np.ones(shape)
    svel = 3000*np.ones(shape)
    density = 2200*np.ones(shape)
    sim = ElasticPSV(pvel, svel, density, spacing=10)
    sim.add_point_source((shape[0]//2, shape[1]//2),
                         dip=45, source=Ricker(5, 10, 1./10))
    sim.run(180)

def test_wavfd_scalar_waveform():
    "wave form comparison "
    shape = (60, 50)
    velocity = 2000*np.ones(shape)
    sim = Scalar(velocity, (50, 50))
    fricker = Ricker(1., 2.5, 1.2/2.5)
    sim.add_point_source((25, 45), fricker)  # far from boundary
    sim.run(100)
    seism = sim[:, 25, 25]
    t = np.linspace(0.0, sim.dt*sim.simsize, sim.simsize)
    analy = analytic.wave2d_analytic(20*50, 2000., sim.dt, fricker(t))
    # normalize everything
    analy = (analy - analy.min())/(analy.max() - analy.min())
    seism = (seism - seism.min())/(seism.max() - seism.min())
    analy -= analy.mean()
    seism -= seism.mean()
    assert np.all(abs(analy-seism) <= 0.03)  # 3 percent amp. diff.

def test_wavefd_elasticsh_run():
    "run and sum two simulation with different phase wavelets that equals zero"
    shape = (50, 50)
    velocity = 1500*np.ones(shape)
    density = 2200*np.ones(shape)
    sim = ElasticSH(velocity, density, (5, 5))
    sim.add_point_source((shape[0]//2, shape[1]//2), Ricker(5, 10.))
    sim.run(100)
    sim_iphase = ElasticSH(velocity, density, (5, 5))
    sim_iphase.add_point_source((shape[0]//2, shape[1]//2), -1*Ricker(5, 10.))
    sim_iphase.run(100)
    diff = sim[-1] + sim_iphase[-1]
    assert np.all(diff <= 0.01), 'diff: %s' % (str(diff))

