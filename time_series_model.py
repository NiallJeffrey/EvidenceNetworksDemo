import numpy as np
from scipy.stats import norm, multivariate_normal
import sys, os, time, gc

def prior_normal(theta, mean_prior=None, cov_prior=None):
    """
    Normal prior evaluation for parameters $\theta$
    :param theta: input parameter values
    :param mean_prior: prior mean for parameters
    :param cov_prior: covariance for parameters
    :return:
    """
    theta = np.array(theta)
    if mean_prior is None:
        mean_prior = np.zeros(theta.shape)
    else:
        mean_prior = np.array(mean_prior)
    if cov_prior is None:
        cov_prior = np.identity(theta.shape[0])
    else:
        cov_prior = np.array(cov_prior)

    return multivariate_normal.pdf(theta, mean=mean_prior, cov=cov_prior)


def coordinate_function(x, j):
    """
    Defines function for linear model matrix: $A_{ij}=f_j(x_i)$
    :param x: data coordinates x
    :param j: parameter index corresponding to: $\theta_j$
    :return: f(x,j) = cos(x * (j + 1/2))
    """
    return np.cos((j - 0.5) * x)


def matrix_operation(x, n_params):
    """
    Generate matrix $A_{ij}$ for linear model
    :param x: data coordinates x
    :param n_params: number of parameters in $\theta$
    :return: matrix $A_{ij}$
    """
    mat_A = np.ones((x.shape[0], n_params))
    for j in np.arange(n_params):
        if j>0:
            mat_A[:, j] = coordinate_function(x, j)
        else:
            mat_A[:, j] = x * 2.
    return mat_A

def mu_data(x, theta, mat_A=None):
    """
    Calculate data mean as function of parameters: $\mu(\theta)$
    :param x: data coordinates x
    :param theta: model parameters $\theta$
    :param mat_A: matrix $A_{ij}$ for linear model
    :return: mean data vector $\mu(\theta)$
    """
    theta = np.array(theta)
    if mat_A is None:
        mat_A = matrix_operation(np.array(x), theta.shape[-1])
    return np.inner(mat_A, theta)


def log_likelihood(theta, x, y, ystd, mat_A=None):
    """
    Gaussian likelihood with diagonal covariance
    :param theta: model parameters $\theta$
    :param x: data coordinates x
    :param y: noisy data y
    :param ystd: standard deviation of data: $\sqrt(\diag(Cov))$
    :param mat_A: matrix $A_{ij}$ for linear model
    :return: log likelihood: $\log L(\theta | y, ystd)$
    """
    mu = mu_data(x, theta, mat_A)
    return -0.5 * np.sum(((y - mu) / ystd) ** 2.) \
           - 0.5 * len(ystd) * np.log(np.pi * 2) \
           - 0.5 * np.sum(np.log(ystd ** 2.))


def evidence_analytic(y, mat_A, mu_prior, cov_prior, cov_data):
    """
    Analytic evidence calculation for model
    :param y: noisy data y
    :param mat_A: matrix $A_{ij}$ for linear model
    :param mu_prior:  prior mean for parameters
    :param cov_prior: prior covariance for parameters
    :param cov_data:  data covariance
    :return:
    """
    mu_evidence = np.dot(mat_A, np.array(mu_prior))
    cov_evidence = np.dot(np.dot(mat_A, np.array(cov_prior)),
                          mat_A.T) + cov_data
    return multivariate_normal.pdf(y, mean=mu_evidence, cov=cov_evidence)


def bayes_factor_theta_0(y, x, n_params, ystd):
    """
    Bayes factor calculation for nested model without $\theta{j=0}=0$
    (logK > 0 implies $\theta_0=0$ is disfavoured)
    :param y: noisy data y
    :param x: data coordinates x
    :param n_params: number of parameters in $\theta$
    :param ystd: standard deviation of data: $\sqrt(\diag(Cov))$
    :return: Bayes factor
    """
    baseline_evidence = evidence_analytic(y, matrix_operation(x, n_params), np.zeros(n_params),
                                          np.identity(n_params), np.identity(len(ystd)) * ystd ** 2)

    alternative_evidence = evidence_analytic(y, matrix_operation(x, n_params)[:, 1:], np.zeros(n_params)[1:],
                                             np.identity(n_params)[1:, 1:], np.identity(len(ystd)) * ystd ** 2)

    return baseline_evidence / alternative_evidence
