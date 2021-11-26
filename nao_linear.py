import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Discretiza o Espaço Amostral de uma Série Temporal para 
# criar uma distribuição de probabilidade discreta
def binning(X, nbins):
  vmin = np.min(X)
  vmax = np.max(X)
  bins = np.linspace(vmin, vmax, nbins)
  bsize = bins[1] - bins[0]
  return bins, bsize

# Busca Binária - Encontra a qual bin um valor x pertence
def binary_search(x, bins, bsize):
  low = 0
  high = len(bins)
  mid = 0
  while low <= high:
    mid = (high + low) // 2
    if bins[mid] + bsize < x:
      low = mid + 1
    elif bins[mid] > x:
      high = mid - 1
    else:
      return bins[mid]
  return -1

# P(X) - Distribuição marginal
def pmf(X, nbins):
  bins, bsize = binning(X, nbins)
  p = { k: 0. for k in bins}
  inc = 1/len(X)
  for x in X:
    p[binary_search(x, bins, bsize)] += inc
  return p

# P(x(t), x(t+tau)) - Distribuição Conjunta
def conj_pmf(X, nbins, tau):
  bins, bsize = binning(X, nbins)
  p = { i: { j: 0. for j in bins } for i in bins}
  inc = 1/(len(X)-tau)
  for i in range(len(X)-tau):
    ii = binary_search(X[i], bins, bsize)
    jj = binary_search(X[i + tau], bins, bsize)
    p[ii][jj] += inc
  return p

# P(X,Y) - distribuição conjunta multivariada
def conj(X, Y, nbins, tau):
  xbins, xbsize = binning(X, nbins)
  ybins, ybsize = binning(Y, nbins)
  p = { i: { j: 0. for j in ybins } for i in xbins}
  inc = 1/(len(X)-tau)
  for i in range(len(X)-tau):
    ii = binary_search(X[i], xbins, xbsize)
    jj = binary_search(Y[i+tau], ybins, ybsize)
    p[ii][jj] += inc
  return p

# Informação Mútua univariada para os lags 0 à tau
def MI(X, tau, nbins=100):
  mi = 0
  p = pmf(X, nbins)
  pij = conj_pmf(X, nbins, tau)
  for i in p.keys():
    for j in p.keys():
      if p[i] > 0 and p[j] > 0 and pij[i][j] > 0:
        mi += pij[i][j] * np.log( pij[i][j] / (p[i] * p[j]) )
  return mi

# Informação Mútua multivariada para os lags 0 à tau
def MI_multivariado(X, Y, nbins=100):
  mi = 0
  px = pmf(X, nbins)
  py = pmf(Y, nbins)
  pij = conj(X, Y, nbins, tau)
  for i in px.keys():
    for j in py.keys():
      if px[i] > 0 and py[j] > 0 and pij[i][j] > 0:
        mi += pij[i][j] * np.log( pij[i][j] / (px[i] * py[j]) )
  return mi

# Entropia de uma distribuição univariada
def H(X, nbins=100):
  h = 0
  p = pmf(X, nbins)
  for i in p.keys():
    if p[i] > 0:
      h += p[i] * np.log( p[i])
  return -h

# Entropia de uma distribuição conjunta
def H_conj(X, Y, nbins=100):
  h = 0
  p = conj(X, Y, nbins, 0)
  for i in p.keys():
    for j in p[i].keys():
      if p[i][j] > 0:
        h += p[i][j] * np.log( p[i][j])
  return -h

# Informação Mútua univariada para os lags 0 à tau
def AMI(X, tau, nbins=100):
  ami = np.zeros(tau)
  p = pmf(X, nbins)
  h = H(X, nbins)
  for t in range(tau):
    mi = 0
    pij = conj_pmf(X, nbins, t)
    for i in p.keys():
      for j in p.keys():
        if p[i] > 0 and p[j] > 0 and pij[i][j] > 0:
          mi += pij[i][j] * np.log( pij[i][j] / (p[i] * p[j]) )
    ami[t] = mi
  return ami/h

# Informação Mútua multivariada para os lags 0 à tau
def AMI_multivariado(X, Y, tau, nbins=100):
  ami = np.zeros(tau)
  px = pmf(X, nbins)
  py = pmf(Y, nbins)
  h = H_conj(X,Y, nbins)
  for t in range(tau):
    mi = 0
    pij = conj(X, Y, nbins, t)
    for i in px.keys():
      for j in py.keys():
        if px[i] > 0 and py[j] > 0 and pij[i][j] > 0:
          mi += pij[i][j] * np.log( pij[i][j] / (px[i] * py[j]) )
    ami[t] = mi
  return ami/h

def plot_ami(X, k, **kwargs):
  ami = AMI(X, k)

  ax = kwargs.get("axis", None)
  if ax is None:
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[10,3])

  ax.hlines([0],[0],[k], color="black")
  ax.vlines([i for i in range(k)], [0 for i in range(k)], ami, color="red")
  ax.scatter([i for i in range(k)], ami, marker="o")
  ax.set_ylim([0,1.05])
  ax.set_xlabel("k")
  ax.set_ylabel("AMI(k)")

def plot_cross_ami(X, Y, k, **kwargs):
  ami = AMI_multivariado(X,Y, k)

  ax = kwargs.get("axis", None)
  if ax is None:
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[10,5])

  ax.hlines([0],[0],[k], color="black")
  ax.vlines([i for i in range(k)], [0 for i in range(k)], ami, color="red")
  ax.scatter([i for i in range(k)], ami, marker="*")
  ax.set_xlabel("k")
  ax.set_ylabel("AMI(k)")
  ax.set_ylim([0,1.05])
  
  
 # Defasagens 
def lags(dados, p):
  T, n = dados.shape
  X = np.zeros((T-p, n*p))
  Y = dados[p:, :]
  for i in range(p, T):
    for j in range(p):
      X[i - p, j*n:(j*n)+n] = dados[i-(p-j), : ]
  return X, Y


def ajustar_nar(dados, p, **kwargs):
  X,Y = lags(dados, p)

  W1, B1, W2, B2, log_erros = MLP.treinar(X, Y, 
                                  ativacao_camada1 = FuncoesAtivacao.identidade,
                                  ativacao_camada2 = FuncoesAtivacao.identidade,
                                  **kwargs)

  plt.plot(log_erros)

  # Previsões in-sample para calcular a variância dos resíduos

  previsoes = nar(dados, p, [W1, B1, W2, B2, None], **kwargs)

  sigma = np.std(Y - previsoes)  

  return W1, B1, W2, B2, sigma


def nar(dados, p, parametros, **kwargs):
  x,_ = lags(dados, p)
  W1, B1, W2, B2,_ = parametros
  y = MLP.regressao(x, W1, B1, W2, B2,
                    ativacao_camada1 = FuncoesAtivacao.identidade,
                    ativacao_camada2 = FuncoesAtivacao.identidade)
  return np.array(y)
  
  
def ajustar_narma(dados, p, q, **kwargs):

  # AR - Prevê a variável endógena

  X,Y = lags(dados, p)

  W11, B11, W12, B12, log_erros = MLP.treinar(X, Y, 
                                  ativacao_camada1 = FuncoesAtivacao.identidade,
                                  ativacao_camada2 = FuncoesAtivacao.identidade,
                                  **kwargs)

  plt.plot(log_erros)

  # MA - Prevê os Resíduos

  previsoes = nar(dados, p, [W11, B11, W12, B12, None], **kwargs)

  residuos = Y - previsoes

  X2,Y2 = lags(residuos, q)

  W21, B21, W22, B22, log_erros = MLP.treinar(X2, Y2, 
                                  ativacao_camada1 = FuncoesAtivacao.identidade,
                                  ativacao_camada2 = FuncoesAtivacao.identidade,
                                  **kwargs)
  
  previsoes2 = nar(residuos, q, [W21, B21, W22, B22, None], **kwargs)

  # AR + MA - Previsão final 

  Yf = previsoes[q:] + previsoes2

  sigma = np.std(Y[q:] - Yf)  

  return W11, B11, W12, B12, W21, B21, W22, B22, sigma

def narma(dados, p, q, parametros, **kwargs):
  x,_ = lags(dados, p)
  W11, B11, W12, B12, W21, B21, W22, B22, _ = parametros
  y1 = MLP.regressao(x, W11, B11, W12, B12,
                    ativacao_camada1 = FuncoesAtivacao.identidade,
                    ativacao_camada2 = FuncoesAtivacao.identidade)
  residuos = dados[p:] - y1
  x2,_ = lags(residuos, q)
  y2 = MLP.regressao(x2, W21, B21, W22, B22,
                    ativacao_camada1 = FuncoesAtivacao.identidade,
                    ativacao_camada2 = FuncoesAtivacao.identidade)
  return np.array(np.array(y1[q:]) + np.array(y2))


def ajustar_nvar(dados, p, **kwargs):
  X,Y = lags(dados, p)

  W1, B1, W2, B2, log_erros = MLP.treinar(X, Y, 
                                  ativacao_camada1 = FuncoesAtivacao.identidade,
                                  ativacao_camada2 = FuncoesAtivacao.identidade,
                                  **kwargs)

  plt.plot(log_erros)

  previsoes = nvar(dados, p, [W1, B1, W2, B2, None], **kwargs)

  residuos = Y - previsoes

  Sigma = np.sqrt(np.cov(residuos, rowvar=False)) 

  return W1, B1, W2, B2, Sigma

def nvar(dados, p, parametros, **kwargs):
  x,_ = lags(dados, p)
  W1, B1, W2, B2,_ = parametros
  y = MLP.regressao(x, W1, B1, W2, B2,
                    ativacao_camada1 = FuncoesAtivacao.identidade,
                    ativacao_camada2 = FuncoesAtivacao.identidade)
  return np.array(y)


def ajustar_nvarma(dados, p, q, **kwargs):
  X,Y = lags(dados, p)

  W11, B11, W12, B12, log_erros = MLP.treinar(X, Y, 
                                  ativacao_camada1 = FuncoesAtivacao.identidade,
                                  ativacao_camada2 = FuncoesAtivacao.identidade,
                                  **kwargs)

  plt.plot(log_erros)

  previsoes = nar(dados, p, [W11, B11, W12, B12, None], **kwargs)

  residuos = Y - previsoes

  X2,Y2 = lags(residuos, q)

  W21, B21, W22, B22, log_erros = MLP.treinar(X2, Y2, 
                                  ativacao_camada1 = FuncoesAtivacao.identidade,
                                  ativacao_camada2 = FuncoesAtivacao.identidade,
                                  **kwargs)
  
  previsoes2 = nar(residuos, q, [W21, B21, W22, B22, None], **kwargs)

  Yf = previsoes[q:, :] + previsoes2

  Sigma = np.sqrt(np.cov(Y[q:, :] - Yf, rowvar=False)) 

  return W11, B11, W12, B12, W21, B21, W22, B22, Sigma

def nvarma(dados, p, q, parametros, **kwargs):
  x,_ = lags(dados, p)
  W11, B11, W12, B12, W21, B21, W22, B22, _ = parametros
  y1 = MLP.regressao(x, W11, B11, W12, B12,
                    ativacao_camada1 = FuncoesAtivacao.identidade,
                    ativacao_camada2 = FuncoesAtivacao.identidade)
  y1 = np.array(y1)
  residuos = dados[p:, :] - y1
  x2,_ = lags(residuos, q)
  y2 = MLP.regressao(x2, W21, B21, W22, B22,
                    ativacao_camada1 = FuncoesAtivacao.identidade,
                    ativacao_camada2 = FuncoesAtivacao.identidade)
  return np.array(y1[q:, :] + np.array(y2))
