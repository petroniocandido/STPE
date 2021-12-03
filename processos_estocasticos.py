import numpy as np    
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

from STPE.distribuicoes import E, Var, Cov


def ACF(X, k, **kwargs):
  c0 = Cov(X, 0, **kwargs)
  ck = Cov(X, k, **kwargs)
  return ck/c0



def plot_acf(X, k, **kwargs):
  acf = []
  for i in range(k):
    acf.append(ACF(X, i, **kwargs))

  ax = kwargs.get("axis", None)
  if ax is None:
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[10,5])

  ax.hlines([0],[0],[k], color="black")
  ax.vlines([i for i in range(k)], [0 for i in range(k)], acf, color="red")
  ax.scatter([i for i in range(k)], acf, marker="*")
  ax.set_xlabel("k")
  ax.set_ylabel("ACF(k)")
  
  
  def comparar_janelas(X):
  m,n = X.shape
  tau = int(n/10)

  fig, ax = plt.subplots(nrows=m, ncols=1,figsize=[10,5])

  indices = range(int((n-tau)/tau))
  for j in range(m):
    ax[j].set_ylabel("Realização {}".format(j))
    data = []
    for i in indices:
      data.append(X[j, i*tau : i*tau + tau])
    ax[j].boxplot(data)

  ax[0].set_title("Janelas de tamanho $\\tau$={}".format(tau))
  plt.tight_layout()
  
  
def comparar_janelas_acf(X):
  m,n = X.shape
  tau = int(n/3)
  indices = range(int(n/tau))

  fig, ax = plt.subplots(nrows=m, ncols=len(indices),figsize=[10,5])

  for j in range(m):
    for i in indices:
      plot_acf(X[:, i*tau : i*tau + tau], 10, modo="realizacao", r=j, axis=ax[j][i])
    ax[j][0].set_ylabel("Realização {}".format(j))

  plt.title("Janelas de tamanho $\\tau$={}".format(tau))
  plt.tight_layout()
  

def processo_bernoulli(n, m, p = 0.5):
  return np.random.binomial(1,p,(m,n))


def ruido_branco(n, m):
  return np.random.randn(m,n)


def passeio_aleatorio(n, m):
  X = ruido_branco(n,m)
  for i in range(m):
    for j in range(1, n):
      X[i,j] += X[i,j-1]
  return X


def processo_wiener(n, m, dt=0.1):
  wgn = ruido_branco(n,m)
  X = np.zeros((m,n))
  sqrt_dt = np.sqrt(dt)
  for i in range(m):
    for j in range(1, n):
      X[i,j] = X[i,j-1] + sqrt_dt*wgn[i,j] 
  return X


def processo_poisson(n, m, l = 0.5):
  T = np.random.exponential(scale=l, size=(m,n))
  for i in range(m):
    for j in range(1, n):
      T[i,j] += T[i,j-1]
  return T


def processo_autogregessivo(n, m, alpha, beta):
  X = np.random.randn(m,n)
  for i in range(m):
    for j in range(1, n):
      X[i,j] += alpha * X[i,j-1] + beta
  return X


def processo_tsp(n, m, alpha, beta):
  X = np.random.randn(m,n)
  for i in range(m):
    for j in range(1, n):
      X[i,j] += alpha * j + beta
  return X


def processo_csp(n, m, P):
  inicial = passeio_aleatorio(P, m)
  X = np.random.randn(m,n)
  for i in range(m):
    for j in range(n):
      if j < P:
        X[i,j] += inicial[i,j]
      else:
        X[i,j] += X[i,j - P]
  return X

def processo_heterocedastico(n, m, alpha):
  X = np.zeros((m,n))
  for i in range(m):
    for j in range(1,n):
      X[i,j] += np.random.normal(loc=0,scale=alpha * j)
  return X
