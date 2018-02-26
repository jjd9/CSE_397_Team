#!/usr/bin/env python
#
import numpy as np
from math import log, pi

def Gaussian(x,sig,mu):
    """
    Calculate normal distribution given parameters
    """
    nom = np.exp((-(x-mu)**2)/(2*sig**2))
    den = sig*np.sqrt(2*np.pi)
    output_G = float(nom)/den
    return output_G

def prior_U(q):
    """
    This routine should return the log of the
    prior probability distribution: P(q|X)
    evaluated for the given value of q.
    Where q is a given value of Uc (exact center line velocity)
    best available knowledge is that Uc = 1.1627
    error expected to be less than 5% with 95% confidence
    """
    mu = 1.1627 #best available information of Uc
    sig = 0.05*mu/1.96
    output_U = Gaussian(q,sig,mu)
    output_U =np.log(output_U)
    return (output_U)

def prior_C(C):
    """
    This routine should return the log of the
    prior probability distribution: P(C|X)
    evaluated for the given value of C.
    Recall C is the discretization error for h = 1.
    no reason to think C is positive or negative
    95% confidence at h=1 that magntiude of C is less than 0.5% of Uc
    """
    mu = 0
    sig = 1.1627*0.005/1.96
    val_prior_C = Gaussian(C,sig,mu)
    output_C = np.log(val_prior_C)

    return (output_C)

def prior_p(p):
    """
    This routine should return the log of the
    prior probability distribution: P(p|X)
    evaluated for the given value of p.
    Recall p is the order of convergence (discretization?).
    p is positive
    value of p is between 1 and 10
    """
    output_P = 0.1
    return np.log(output_P)

#
# One should not have to edit the routine below
#
def prior(q,C,p):
    """
    This routine should return the log of the
    prior probability distribution: P(q,C,p|X)
    evaluated for the given values of q, C, p.
    """

    # for some reason the p guesses are sometimes negative, this
    # patches that up
    if(p < 0):
        return -1 * np.inf

    return prior_U(q) + prior_C(C) + prior_p(p)
