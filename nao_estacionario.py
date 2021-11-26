import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
#   AR(1) VARIANTE NO TEMPO 
#----------------------------------------------------------------------------

def tv_ar1(dados, beta):
  n = len(dados)
  alfas = np.ones(n+1)
  alfa = 1.
  ret = np.zeros(n)
  ret[0] = dados[0]
  for i in range(2, n):
    ret[i] = dados[i-1] * alfa
    alfas[i] = dados[i-1]/dados[i-2] if dados[i-2] > 0 else 0 
    alfa = beta * alfas[i] + (1 - beta) * alfa
  
  return ret[2:]

#----------------------------------------------------------------------------
#   ARMA(1,1) VARIANTE NO TEMPO 
#----------------------------------------------------------------------------

def tv_arma11(dados, gamma, delta):
  previsoes_ar = tv_ar1(dados, gamma)
  residuos_ar = dados[2:] - previsoes_ar
  previsoes_ma = tv_ar1(residuos_ar, delta)
  return previsoes_ar[2:] + previsoes_ma

#----------------------------------------------------------------------------
#   AR VARIANTE NO TEMPO COM FILTRO DE KALMAN 
#----------------------------------------------------------------------------


def kalman_predicao(a, V, F, G, tau):
  a_t = F @ a
  V_t = F @ V @ F.T + (tau * G @ G.T)

  return a_t, V_t


def kalman_filtragem(a, V, y, H, sigma):
  p = len(a)
  K =  V @ H.T * (H @ V @ H.T + sigma)**-1
  res_y = y - (H @ a)
  a_t = a + K * res_y
  V_t = (np.eye(p) - K @ H) @ V

  return a_t, V_t


def suavizacao(a, a_t, V, V_t, alpha ):
  a_f = alpha * a + (1 - alpha) * a_t
  V_f = alpha * V + (1 - alpha) * V_t
  return a_f, V_f


def tv_ar_kalman(dados, p, alfa=.9):
  n = len(dados)
  
  ## INICIALIZAÇÃO

  # Vetor de Estado / Coeficientes
  a = np.random.rand(p)
  a_t = a

  # Matrizes de Covariância
  V = np.eye(p)
  V_t = V
  F = np.eye(p)
  G = np.eye(p)

  # Resíduos e Variâncias
  res_y = np.ones(n) 
  sigma = 1.
  res_a = np.ones(n) 
  tau = 1.


  ret = np.zeros(n)
  ret[:p] = dados[:p]
  H = dados[0:p] 
  
  for i in range(p, n):

    # PREDIÇÃO DE ŷ(t+1)

    H_t = H
    H = dados[i-p:i]  # Vetor de defasagens / lags
    y = a @ H
    ret[i] = y

    #ATUALIZAÇÃO DOS COEFICIENTES USANDO O FILTRO DE KALMAN

    a_t = a
    V_t = V
    a, V = kalman_predicao(a_t, V_t, F, G, tau)
    a, V = kalman_filtragem(a, V, dados[i-1], H_t, sigma)
    a, V = suavizacao(a, a_t, V, V_t, alfa)
    
    # Resíduos de Y
    res_y[i] = dados[i] - y
    sigma = np.std(res_y)

    #Resíduos de a
    res_a[i] = np.mean(a_t - a)
    tau = np.std(res_a)

  return ret[p:]

def tv_arma_kalman(dados, p, q, alfa=.9, beta=.1):
  estimativas_ar = tv_ar_kalman(dados, p, alfa)
  residuos_ar = dados[p:] - estimativas_ar
  estimativas_ma = tv_ar_kalman(residuos_ar, q, beta)
  return estimativas_ar[q:] + estimativas_ma
