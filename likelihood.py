#!/usr/bin/env python
#
import numpy as np
import prior
from math import log
#a
def likelihood(q,C,p):
    """
    This routine should return the log of the
    likelihood function: P(qi|q,C,p,X)
    evaluated for given values of q, C and p
    """
    #------------------------------------
    val_likelihood = 1
    #Run through each mesh
    h = [1, 0.5, np.sqrt(2), 2.0]
    sig = [0.0001282, 0.0002336, 0.0003982, 0.0001283]
    mu = [1.16367827195, 1.16389876649, 1.16429173392, 1.16828362427 ]
    for i, val in enumerate(h):
        Uc = q - C*val**p
        val_likelihood *= prior.Gaussian(Uc,sig[i],mu[i])
    return log(val_likelihood)
