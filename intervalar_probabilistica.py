import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

def plot_prob(y, mu, sigma, ax):
  x = np.linspace(mu-3*sigma, mu+3*sigma, 35)
  probs = [norm.pdf(k, loc=mu, scale=sigma) for k in x]
  ax.vlines([y],[0],[np.max(probs)*1.3], color='red')
  ax.vlines([mu],[0],[np.max(probs)*1.3], color='blue')
  ax.plot(x,probs)
  
def intervalo_interquantil(x, y, alfa, sigma, cor, ax):
  sup = [norm.ppf(1 - alfa/2, loc=k, scale=sigma) for k in y]
  inf = [norm.ppf(alfa/2, loc=k, scale=sigma) for k in y]
  ax.plot(x, inf, c=cor, label='$\\alpha=${}'.format(alfa))
  ax.plot(x, sup, c=cor)
  
def ajustar_arch(residuos, parametros):
  r2 = residuos ** 2
  param = ajustar_ar(r2,parametros)
  return param

def arch(residuos, parametros):
  ordem = len(parametros[0])
  sigma = ar(residuos, parametros)
  return sigma
