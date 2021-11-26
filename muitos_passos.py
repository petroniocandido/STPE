

def lags_H(dados, p, H=1):
  n = len(dados)
  X = np.zeros((n-p-H, p))
  Y = dados[p+H:]
  for i in range(p, n-H):
    X[i-p,:] = dados[i-p:i]
  return X,Y


def ajustar_ar_H(dados, parametros, H = 1):
  p = parametros[0]
  X,Y = lags_H(dados, p, H)
  coef = np.linalg.inv(X.T.dot(X)).dot( X.T.dot(Y) )

  previsoes = ar(dados, [coef, None])
  residuos = dados[p+H:] - previsoes[:-H]
  sigma = np.std(residuos)

  return coef, sigma

def ar_recursivo(dados, parametros, H=1):
  coef, sigma = parametros
  p = len(coef)
  n = len(dados)
  ndados = np.zeros(n+H+1)
  ndados[:n] = dados
  ret = np.zeros(n-p+H)
  for i in range(p,n+H):
    ret[i-p] = np.array(ndados[i-p:i]).dot(coef)
    ndados[i+1] = ret[i-p]

  return ret

def sigma_H(sigma, t, beta):
  return sigma * (t * beta)
