import numpy as np    
from scipy import stats
import matplotlib.pyplot as plt


def phi(px):
  rads = np.linspace(-np.pi, np.pi, 100)
  ret = { w : np.sum([px[x] * np.exp(w*1j*x) for x in px.keys()]) for w in rads}
  return ret


def phi_plot(px, ax):
  fphi = phi(px)
  ax.plot([k for k in fphi.keys()], [k for k in fphi.values()])
  ax.set_xlabel("$\omega$")
  ax.set_ylabel("$\phi(\omega)$")
  

def momento(px, n):
  ret = 0
  for x in px.keys():
    ret += (x ** n) * px[x]
  return ret


def momento_central(px, n):
  mu = momento(px, 1)
  ret = 0
  for x in px.keys():
    ret += (x - mu) ** n * px[x]
  return ret


def momento_normalizado(px, n):
  mu = momento(px, 1)
  sigma = momento_central(px, 2)
  ret = 0
  for x in px.keys():
    ret += ((x - mu)/sigma) ** n * px[x]
  return ret
