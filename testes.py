from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene
from statsmodels.stats.diagnostic import acorr_ljungbox

def ADF(X):
  print("\n\t---------------------------------------------------------")
  print("\tTeste de Estacionariedade - Dickey-Fuller Aumentado (ADF)")
  print("\t---------------------------------------------------------")
  print("H0: O processo é não-estacionário.")
  print("H1: O processo é estacionário.")
  result = adfuller(X)
  print('\nEstatística ADF: {}'.format(result[0]))
  print('p-Valor: {}'.format(result[1]))
  print('Valores Críticos:')
  print('\tAlfa\tVal. Crit\tResultado')
  for key, value in result[4].items():
    h0 = "H0 Aceita" if result[0] > value else "H0 Rejeitada"
    print('\t{}\t{}\t{}'.format(key, value, h0))
    
def Levene(X):
  print("\n\t---------------------------------------------------------")
  print("\tTeste de Homocedasticidade - Levene")
  print("\t---------------------------------------------------------")
  print("H0: As variâncias das sub-amostras são iguais, a série é homocedástica.")
  print("H1: As variâncias das sub-amostras são diferentes, a série é heterocedástica")
  janelas = np.array_split(X, 5)
  resultado = levene(*janelas)
  print("\nEstatística de teste: {}".format(resultado.statistic))
  print("p-Valor: {}".format(resultado.pvalue))
  print("Resultado: {}".format('H0 Aceita' if resultado.pvalue > 0.05 else 'H0 Rejeitada'))
  
def LjungBox(X, lags=12):
  print("\n\t---------------------------------------------------------")
  print("\tTeste de Autocorrelação - Ljung-Box")
  print("\t---------------------------------------------------------")
  print("H0: O lag não tem autocorrelação (é IID).")
  print("H1: O lag tem autocorrelação.")
  lj, pval = acorr_ljungbox(X, lags=lags)
  for i in range(len(lj)):
    print("LAG {}".format(i+1))
    print("\tEstatística de teste: {}".format(lj[i]))
    print("\tp-Valor: {}".format(pval[i]))
    print("\tResultado: {}".format('H0 Aceita' if pval[i] > 0.05 else 'H0 Rejeitada'))
