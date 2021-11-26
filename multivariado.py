import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------
#   MÉTRICAS
#----------------------------------------------------------------------------


def AutoCov(X, k):
  n = len(X)
  mx = np.mean(X)
  c = np.zeros(n-k)
  for i in range(n - k):
    c[i] = (X[i] - mx)*(X[i+k] - mx)
  c = c.mean()
  
  return c


def CrossCov(X, Y, k):
  n = len(X)
  mx = np.mean(X)
  my = np.mean(Y)
  c = np.zeros(n-k)
  for i in range(n - k):
    c[i] = (X[i] - mx)*(Y[i+k] - my)
  c = c.mean()
  
  return c


def CCF(X, Y, k):
  gamma_xy = CrossCov(X, Y, k)
  gamma_x = Var(X)
  gamma_y = Var(Y)
  return gamma_xy / np.sqrt(gamma_x * gamma_y)


def plot_ccf(X, Y, k, **kwargs):
  ccf = []
  for i in range(k):
    ccf.append(CCF(X, Y,  i))

  ax = kwargs.get("axis", None)
  if ax is None:
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[10,5])

  ax.hlines([0],[0],[k], color="black")
  ax.vlines([i for i in range(k)], [0 for i in range(k)], ccf, color="red")
  ax.scatter([i for i in range(k)], ccf, marker="*")
  ax.set_xlabel("k")
  ax.set_ylabel("CCF(k)")


#----------------------------------------------------------------------------
#   MODELOS COM VARIÁVEIS EXÓGENAS
#----------------------------------------------------------------------------
  
  
def lags_x(endog, exog, k):
  t = len(endog)
  n = len(exog)
  X = np.zeros((t-k, (n+1)*k))
  Y = endog[k:]
  for i in range(k, t):
    X[i-k,0:k] = endog[i-k:i]
    for e in range(n):
      X[i-k,(e+1)*k:(e+2)*k] = exog[e][i-k:i]
  return X,Y


def ajustar_arx(endog, exog, parametros):
  p = parametros[0]
  X,Y = lags_x(endog, exog, p)

  coef = np.linalg.inv(X.T @ X) @ (X.T @ Y )

  previsoes = arx(endog, exog, [coef, None])
  
  residuos = endog[p:] - previsoes

  sigma2 = np.std(residuos)

  return coef, sigma2 


def arx(endog, exog, parametros):
  coef, _ = parametros
  t = len(endog)
  n = len(exog)
  p = int(len(coef) / (n+1))
  ret = np.zeros(t-p)
  X, _ = lags_x(endog, exog, p)
  for i in range(p,t):
    ret[i-p] = X[i-p].dot(coef)

  return ret

#----------------------------------------------------------------------------
#   MODELOS COM MULTIVARIADOS / VETORIAIS
#----------------------------------------------------------------------------


def lags_v(dados, p):
  T, n = dados.shape
  X = np.zeros((T-p, n*p))
  Y = dados[p:, :]
  for i in range(p, T):
    for j in range(p):
      X[i - p, j*n:(j*n)+n] = dados[i-(p-j), : ]
  return X, Y

def var(dados, parametros):
  T, n = dados.shape
  coef, _ = parametros
  p = int(coef.shape[0]/n)
  X,_ = lags_v(dados, p)
  ret = np.zeros((T-p, n))
  for i in range(T-p):
    ret[i, :] = coef.T @ X[i, :] 
  return ret 

def ajustar_var(dados, p):
  X,Y = lags_v(dados, p)
  
  coef = np.linalg.inv(X.T @ X) @ ( X.T @ Y )

  previsoes = var(dados, [coef, None])

  residuos = dados[p:, :] - previsoes

  Sigma = np.sqrt(np.cov(residuos, rowvar=False))

  return coef, Sigma
