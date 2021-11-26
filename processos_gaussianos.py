import numpy as np
import matplotlib.pyplot as plt

def simular_gp(mu, sigma, m, ax=None):
  X = np.random.multivariate_normal(mu, cov=sigma, size=m)
  d = mu.shape[0]

  if ax is None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,5))

  for i in range(m):
    ax.plot(X[i,:])
  ax.set_xlabel("T")
  ax.set_ylabel("$X_i$")
  ax.set_xticks([i for i in range(d)])

def m_zeros(X, parametros):
  return np.zeros(X.shape)

def m_constante(X, parametros):
  c = parametros[0]
  return np.repeat(c, X.shape[0]).reshape(X.shape)

def m_linear(X, parametros):
  a, b = parametros
  return np.linspace(a,b, X.shape[0]).reshape(X.shape)

def Cov(x1, x2, kernel, parametros):
  n1 = x1.shape[0]
  n2 = x2.shape[0]
  cov = np.zeros((n1,n2))
  for i in range(n1):
    cov[i] = kernel(x1[i], x2, parametros)
  return cov

def plotar_cov(cov, m):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
  d = cov.shape[0]
  map = ax[0].matshow(cov)
  ax[0].set_xlabel("$X_1$")
  ax[0].set_ylabel("$X_2$")
  ax[0].set_title("$\Sigma$")
  fig.colorbar(map, ax=ax[0])
  simular_gp(m_zeros(x1, []).flatten(), cov, m, ax=ax[1])
  ax[1].set_title("Amostras do $\mathcal{GP}$")
  plt.tight_layout()

def plotar_kernel(x1, x2, kernel, parametros, mu=m_zeros, mu_parametros=[], m=5):
  cov = Cov(x1, x2, kernel, parametros)
  plotar_cov(cov, m)
  
def k_linear(x1, x2, parametros):
  return x1.dot(x2.T)

def k_polinomial(x1, x2, parametros):
  return x1.dot(x2.T) ** parametros[0]

def k_rbf(x1, x2, parametros):
  sigma = parametros[0]
  return np.exp(-.5*sigma**2 * np.sqrt(np.sum((x1 - x2)**2, axis=1))).flat

def k_laplaciano(x1, x2, parametros): 
  alpha = np.array(parametros[0])
  return np.exp(-alpha * np.abs(x1 - x2)).flat

def sin2(x):
  return (1 - np.cos(2*x))/2

def k_periodico(x1, x2, parametros):
  p = np.array(parametros[0])
  sigma = np.array(parametros[1])
  l = np.array(parametros[2])
  return sigma*np.exp(-(2 * sin2((np.pi * np.abs(x1 - x2))/p)/l)).flat

from numpy import linalg

def GP(Tc, Xc, Td, mu, kernel, mu_parametros=[], kernel_parametros=[], ruido=0):
  Scc = Cov(Tc, Tc, kernel, kernel_parametros) + np.eye(Xc.shape[0])*ruido
  Sdd = Cov(Td, Td, kernel, kernel_parametros)
  Scd = Cov(Tc, Td, kernel, kernel_parametros)
  Sdc = Scd.T
  Xd = mu(Td, mu_parametros) + Sdc.dot( linalg.inv(Scc)).dot(Xc - mu(Tc, mu_parametros))
  Sd = Sdd - Sdc.dot(linalg.inv(Scc)).dot(Scd)
  return Xd, Sd
