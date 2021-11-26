import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy.stats import probplot
import pylab 

def mse(originais, previsoes):
  return np.sum( (originais - previsoes) ** 2 ) / len(originais)

def rmse(originais, previsoes):
  return np.sqrt( mse(originais, previsoes) ) 

def nrmse(originais, previsoes):
  amplitude = np.max(originais) - np.min(originais)
  return rmse(originais, previsoes) / amplitude

def mape(originais, previsoes):
  return (np.sum( np.abs(originais - previsoes) / np.abs(originais) ) / len(originais)) * 100

def u(originais, previsoes):
  n = len(originais)
  ret = 0
  for i in range(1, n):
    ret += np.abs(originais[i] - previsoes[i]) / np.abs(originais[i] - originais[i-1])
  return ret/n

def r2(originais, previsoes):
  mx = np.mean(originais)
  num = np.sum( (originais - previsoes) ** 2 )
  den = np.sum( (originais - mx) ** 2 )
  
  return 1 - num/den

def mde(originais, previsoes):
  a = np.sign( originais[1:] - previsoes[:-1] )
  b = np.sign( originais[:-1] - originais[1:] )
  return np.sum(np.where(a == b, 1, 0)) / len(originais)

def medir(originais, previsoes, ordem):
  return pd.DataFrame([[mse(originais[ordem:], previsoes), rmse(originais[ordem:], previsoes), \
                       nrmse(originais[ordem:], previsoes), mape(originais[ordem:], previsoes), \
                       u(originais[ordem:], previsoes), r2(originais[ordem:], previsoes), 
                       mde(originais[ordem:], previsoes)]], \
                      columns=['MSE','RMSE','nRMSE','MAPE','U','R2','MDE'] )


def analise_residuos(originais, previsoes, ordem):
  
  residuos = originais[ordem:] - previsoes

  fig, ax = plt.subplots(1, 3, figsize=(15, 5))

  # ACF
  ax[0].plot(residuos)
  ax[0].set_title("Resíduos")

  # ACF
  plot_acf(residuos, lags=20, ax=ax[1])
  ax[1].set_title("ACF")
  
  # Q-Q Plot
  probplot(residuos, dist="norm", plot=ax[2])
  ax[2].set_title("Quantil-Quantil")

  plt.tight_layout()
