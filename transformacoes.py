import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def media_movel(Y, k=3):
  return np.array([np.mean(Y[i-k:i]) for i in range(k,len(Y))])


def binning(X, nbins=10):
  bins = np.linspace(np.min(X), np.max(X), nbins)
  indices = np.digitize(X, bins)
  return np.array([bins[i-1] for i in indices])


def remover_media(Y):
  m = np.mean(Y)
  return Y - m


def normalizar(Y, inversa=False):
  mu = np.mean(Y)
  sigma = np.std(Y)
  return (Y - mu)/sigma


def escala(Y):
  _min = np.min(Y)
  _max = np.max(Y)
  return (Y - _min)/ (_max - _min)


def diferenciar(Y, ordem=1):
  tmp = Y
  for i in range(ordem):
    tmp2 = [tmp[i-1] - tmp[i] for i in range(1, len(tmp))]
    tmp = tmp2
  return tmp


def remover_tendencia(Y, grau=2):
  T = np.array([i for i in range(len(Y))])
  regressao = np.poly1d(np.polyfit(T, Y, grau))

  return np.array([Y[i] - regressao(i) for i in T]).flatten()


def remover_sazonalidade(Y, periodo=10):
  return np.array([Y[i] - Y[i - periodo] for i in range(periodo, len(Y))])


def estabilizar_variancia(Y, grau=2, janela=12):
  T = np.array([i for i in range(len(Y))])
  Y2 = np.repeat(np.sqrt(np.std(Y[0:janela])), len(Y))
  for i in range(janela, len(Y)):
    Y2[i] += np.sqrt(np.std(Y[i-janela: i]))

  regressao = np.poly1d(np.polyfit(T, Y, grau))

  return np.array([Y[i] / regressao(i) for i in T]).flatten()


def box_cox(Y, l=.1):
  ret = np.zeros(len(Y))
  for i in range(len(Y)):
    if Y[i] != 0:
      ret[i] = (Y[i] ** l - 1)/l
    else:
      ret[i] = np.log(Y[i])
  return ret


def roi(Y):
  return np.array([(Y[i-1] - Y[i])/Y[i-1] for i in range(1,len(Y))])
