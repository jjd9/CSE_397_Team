#!/usr/bin/env python
#
import numpy as np
from math import log, pi
def Gaussian(x,sig,mu):
    """
    Calculate normal distribution give parameters
    """
    nom = np.exp((-(x-mu)**2)/(2*sig**2))
    den = sig*np.sqrt(2*np.pi)
    if den == 0:
        output = 0
    else:
        output = nom/den
    return output

#a
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
    val_f = Gaussian(q,sig,mu)
    #if val_f == 0:
    #    val_f = 1
    val_f =np.log(val_f)
    return (val_f)

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
    #if val_prior_C == 0:
    #    val_prior_C = 1
    output = np.log(val_prior_C)
    return (val_prior_C)

def prior_p(p):
    """
    This routine should return the log of the
    prior probability distribution: P(p|X)
    evaluated for the given value of p.
    Recall p is the order of convergence (discretization?).
    p is positive
    value of p is between 1 and 10
    """
    val_prior_p = 0.1
    return np.log(val_prior_p)

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
