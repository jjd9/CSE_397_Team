#!/usr/bin/env python
import numpy as np
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

DESCRIPTION = 'Script to ensure all libraries installed properly'


def lnprob(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)

ndim, nwalkers = 5, 100
ivar = 1. / np.random.rand(ndim)
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[ivar])
sampler.run_mcmc(p0, 1000)

for i in range(ndim):
    pl.figure()
    pl.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
    pl.title("Dimension {0:d}".format(i))
    name = 'example_%i.png' % int(i)
    pl.savefig(name)
print('exiting successfully')
