import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def naive(dados, parametros):
  ''' Método Naïve: ŷ(t+1) = y(t) '''
  ret = np.zeros(len(dados)-1)  
  ret = dados[:-1]
  return ret


def lags(dados, p):
  ''' Dado um conjunto de dados, calcula as matrizes X e Y de defasagens de um processo autoregressivo de ordem p '''
  n = len(dados)
  X = np.zeros((n-p, p))
  Y = dados[p:]
  for i in range(p, n):
    X[i-p,:] = dados[i-p:i]
  return X,Y


def ajustar_ar(dados, parametros):
  '''
  AR(p) - Função de estimação de parâmetros.
  Usando o método de Mínimos Quadrados Ordinários, ajusta os coeficientes de um processo autoregressivo de ordem p. 
  '''
  p = parametros[0]
  X,Y = lags(dados, p)
  coef = np.linalg.inv(X.T.dot(X)).dot( X.T.dot(Y) )
  return coef 


def ar(dados, coef):
  '''
  AR(p) - Função de inferência.
  Estima os próximos valores de um processo autoregressivo de ordem p usando coeficientes dados
  '''
  p = len(coef)
  n = len(dados)
  ret = np.zeros(n-p)
  for i in range(p,n):
    ret[i-p] = dados[i-p:i].dot(coef)

  return ret


def ajustar_arma(dados, parametros):
  ''''
  ARMA(p,q) - Função de estimação de parâmetros.
  Usando o método de Mínimos Quadrados Ordinários, ajusta os coeficientes de um processo autoregressivo 
  de ordem p com médias móveis de ordem q. 
  '''
  p, q = parametros

  # Ajusta o Modelo AR(p)
  alfa = ajustar_ar(dados, [p])

  # Gera as estimativas usando o AR(p)
  previsoes = ar(dados, alfa)
  
  # Resíduos do modelo AR(p)
  residuos_ar = dados[p:] - previsoes
  
  # Ajusta o Modelo MA(q)
  beta = ajustar_ar(residuos_ar, [q])

  # Gera as estimativas dos resíduos usando o MA(q)
  previsoes_ma = ar(residuos_ar, beta)

  # Resíduos finais
  residuos_ma = residuos_ar[q:] - previsoes_ma

  # Calcula o desvio padrão dos resíduos finais
  sigma = np.std(residuos_ma)

  return alfa, beta, sigma


def arma(dados, parametros):
  ''''
  ARMA(p,q) - Função de inferência.
  Estima os próximos valores de um processo autoregressivo de ordem p com médias móveis de ordem q usando coeficientes dados 
  '''
  alfa, beta, sigma = parametros
  p = len(alfa)
  q = len(beta)

  previsoes_ar = ar(dados, alfa)
  residuos_ar = dados[p:] - previsoes_ar

  previsoes_ma = ar(residuos_ar, beta)
  return previsoes_ar[q:] + previsoes_ma


def diferenciar(dados, ordem=1):
  tmp = dados
  for i in range(ordem):
    tmp2 = [tmp[i-1] - tmp[i] for i in range(1, len(tmp))]
    tmp2.insert(0,0)
    tmp = tmp2
  return np.array(tmp)


def integrar(dados, ordem=1, inicial = 0):
  tmp = dados
  tmp[0] = inicial

  for i in range(ordem):
    tmp2 = np.zeros(len(dados))
    for i in range(1, len(tmp)):
      tmp2[i] = tmp[i-1] + tmp[i] 
    tmp = tmp2
  return tmp


def ajustar_arima(dados, parametros):
  ''''
  ARIMA(p,d,q) - Função de estimação de parâmetros.
  Usando o método de Mínimos Quadrados Ordinários, ajusta os coeficientes de um processo autoregressivo 
  de ordem p, com diferenciação integrada de ordem d e médias móveis de ordem q. 
  '''
  p, d, q = parametros
  dados_diff = diferenciar(dados, ordem=d)
  alfa = ajustar_ar(dados_diff, [p])
  previsoes_diff = ar(dados_diff, alfa)
  previsoes1 = integrar(previsoes_diff, d, inicial=dados[p]) 

  residuos1 = dados[p:] - previsoes1

  beta = ajustar_ar(residuos1, [q])
  previsoes_r = ar(residuos1, beta)
  residuos2 = residuos1[q:] - previsoes_r
  sigma = np.std(residuos2)
  return alfa, beta, d, sigma


def arima(dados, parametros):
  ''''
  ARIMA(p,d,q) - Função de inferência.
  Estima os próximos valores de um processo autoregressivo de ordem p, com diferenciação integrada 
  de ordem d e médias móveis de ordem q usando coeficientes dados 
  '''
  alfa, beta, d, sigma = parametros
  n = len(dados)
  p = len(alfa)
  q = len(beta)
  dados_diff = diferenciar(dados, d)
  previsoes_diff = ar(dados_diff, alfa)
  previsoes1 = integrar(previsoes_diff, d, inicial=dados[p])
  residuos1 = dados[p:] - previsoes1
  previsoes_res = ar(residuos1, beta)
  previsoes2 = previsoes1[q:] + previsoes_res 
  return previsoes2


def es(Y, parametros):
  ''' Alisamento Esponencial (Exponential Smoothing) de ordem p e expoente alfa '''
  alfa, ordem = parametros
  n = len(Y)
  ret = np.zeros(n - ordem)
  for i in range(ordem, n):
    ret[i-ordem] = np.sum([alfa * (1 - alfa)** j * Y[i - j - 1] for j in range(0, ordem)])
  return ret


def holt_winters(dados, parametros):
  alfa, beta, gamma, m = parametros
  n = len(dados)
  L = np.zeros(n)
  L[:m+1] = dados[:m+1]
  T = np.zeros(n)
  T[:m+1] = diferenciar(dados[:m+1], 1)
  S = np.zeros(n)
  S[:m+1] = dados[:m+1]
  for t in range(m, n):
    L[t] = alfa * (dados[t-1] - S[t-m]) + (1 - alfa) * (L[t-1] + T[t-1])
    T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
    S[t] = gamma * (dados[t-1] -  L[t] - T[t-1]) + (1 - gamma) * S[t-m]

  return L[m:] + T[m:] + S[m:]
