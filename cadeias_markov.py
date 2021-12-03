import numpy as np
import matplotlib.pyplot as plt


def P_cumulativa(pi):
  pif = [pi[0]]
  for i in range(1, len(pi)):
    pif.append( pif[-1] + pi[i])
  return pif


def escolhe_estado(S, fpi):
  r = np.random.rand(1)[0]
  for i in range(len(S)):
    if fpi[i] >= r:
      return S[i]

    
def simular_cadeia_markov(S, pi, P, n, m):
  processo = np.zeros((m,n))
  for j in range(m):
    pi_t = pi
    pif = P_cumulativa(pi_t)
    processo[j, 0] = escolhe_estado(S, pif)
    for i in range(1, n):
      p = P[ int(processo[j, i - 1]) , : ]
      pf = P_cumulativa(p)
      processo[j, i] = escolhe_estado(S, pf)
  return processo

def mat_pot(mat, n):
  return np.linalg.matrix_power(mat, n)

def e_regular(mat, n):
  return np.all(mat_pot(mat,n) > 0)

def transicao(pi, P, n):
  return pi.dot(mat_pot(P, n))

def simular_convergencia(S, pi, P, n, nomes=None):
  pit = pi
  ns = pi.shape[0]
  pis = np.zeros((n,ns))
  for i in range(n):
    pit = pit.dot(P)
    pis[i,:] = pit
  for i in range(ns):
    plt.plot(pis[:,i], label="{}".format(i if nomes is None else nomes[i]))
  plt.legend()
  plt.tight_layout()
  
def dist_estacionaria(S, P):
  m = len(S)
  A = np.append(P.T - np.identity(m), np.ones((1,m)),axis=0)
  b = np.zeros(m+1)
  b[-1] = 1
  b = b.T
  return np.linalg.solve(A.T.dot(A), A.T.dot(b))
