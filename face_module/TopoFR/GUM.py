### GUM model
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import preprocessing
import torch


def gauss_unif(x):
    global delta, pi, Sigma
    N, D = x.shape

    x=x.astype(float)

    x=preprocessing.scale(x)

    ###initialization
    pi = 0.1
    Mu = np.zeros([1,D])
    epsilon = x - Mu
    Sigma =1 / N *np.sum(epsilon.reshape(-1, D, 1) * epsilon.reshape(-1, 1, D), axis=0)


    c = 0.2

    log_like0 = 0
    for i in range(1000):
        ###E-step
        phi=multivariate_normal(Mu,Sigma+1e-3).pdf(x)
        gamma = phi*pi / (phi*pi + (1 - pi) * c)

        ####M-step
        N1 = gamma.sum()
        #Mu = 1 / N1 * np.sum(gamma.reshape(-1, 1) * x, axis=0)
        epsilon = x - Mu
        Sigma = 1 / N1 * np.sum(gamma.reshape(-1, 1, 1) * epsilon.reshape(-1, D, 1) * epsilon.reshape(-1, 1, D), axis=0)
        pi_new = N1 / N
        thres=0.5
        if pi_new<=thres:
            pi=pi_new
        else:
            pi=thres

        C1 = 1 / N1 * np.sum((1 - gamma.reshape(-1, 1)) / (1 - pi + 1e-10) * epsilon, axis=0)
        C2 = 1 / N1 * np.sum((1 - gamma.reshape(-1, 1)) / (1 - pi + 1e-10) * epsilon ** 2, axis=0)
        if abs(3 * C2 - C1 ** 2)<0.001:
            c=0.001
        else:
            c = 1.0 / (np.prod(2 * np.sqrt(3 * C2 - C1 ** 2)) + 1e-10)
        #c=1/(np.max(np.abs(epsilon))-np.min(np.abs(epsilon)))


        ###stopping criterion
        log_like = np.sum(pi * phi + (1 - pi) * c)
        if abs(log_like - log_like0) < 1e-10:
            break
        else:
            log_like0 = log_like

    delta = c
    return gamma,pi,Sigma[0,0]

