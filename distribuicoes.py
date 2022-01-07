import numpy as np    
from scipy import stats
import matplotlib.pyplot as plt


def phi(px):
  ''' Função característica da PMF P(x) '''
  rads = np.linspace(-np.pi, np.pi, 100)
  ret = { w : np.sum([px[x] * np.exp(w*1j*x) for x in px.keys()]) for w in rads}
  return ret


def phi_plot(px, ax):
  fphi = phi(px)
  ax.plot([k for k in fphi.keys()], [k for k in fphi.values()])
  ax.set_xlabel("$\omega$")
  ax.set_ylabel("$\phi(\omega)$")
  

def momento(px, n):
  ''' Gera o momento de n-enésima ordem da PMF P(x) '''
  ret = 0
  for x in px.keys():
    ret += (x ** n) * px[x]
  return ret


def momento_central(px, n):
  ''' Gera o momento central de n-enésima ordem da PMF P(x) '''
  mu = momento(px, 1)
  ret = 0
  for x in px.keys():
    ret += (x - mu) ** n * px[x]
  return ret


def momento_normalizado(px, n):
  ''' Gera o momento central normalizado de n-enésima ordem da PMF P(x) '''
  mu = momento(px, 1)
  sigma = momento_central(px, 2)
  ret = 0
  for x in px.keys():
    ret += ((x - mu)/sigma) ** n * px[x]
  return ret

def E(X, **kwargs):
  ''' Calcula o valor esperado da PMF P(x) '''
  m,n = X.shape
  e = 0.0
  modo = kwargs.get("modo", "realizacao") # tempo, realizacao, ensemble
  if modo == "tempo":
    t = kwargs.get("t", 0)
    e = X[:, t].mean()
  elif modo == "realizacao":
    r = kwargs.get("r", 0)
    e = X[r, :].mean()
  else:
    e = X.mean()
  return e

def Var(X, k, **kwargs):
  ''' Calcula a variância da PMF P(x) '''
  m,n = X.shape
  mx = E(X, **kwargs)
  v = 0.0
  modo = kwargs.get("modo", "realizacao") # tempo, realizacao, ensemble
  if modo == "tempo":
    t = kwargs.get("t", 0)
    v = np.mean( (X[:, t] - mx)**2 )
  elif modo == "realizacao":
    r = kwargs.get("r", 0)
    v = np.mean( (X[r, :] - mx)**2 )
  else:
    v = np.mean( (X - mx)**2 )
  return v

def Cov(X, k, **kwargs):
  ''' Calcula a autocovariância do processo estocástico X para a defasagem k '''
  
  m,n = X.shape
  modo = kwargs.get("modo", "realizacao")
  mx = E(X, **kwargs)
  
  if modo == "realizacao":
    c = np.zeros(n-k)
    r = kwargs.get("r", 0)
    for i in range(n - k):
      c[i] = (X[r,i] - mx)*(X[r,i+k] - mx)
  else:
    c = np.zeros((m, n-k))
    for r in range(m):
      for i in range(n - k):
        c[r, i] = (X[r,i] - mx)*(X[r,i+k] - mx)

  c = c.mean()
  
  return c
