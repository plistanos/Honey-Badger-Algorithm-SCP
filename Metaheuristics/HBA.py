import random
import time
import numpy as np
rng = np.random.default_rng()
import math
import sys
from numpy import linalg as LA
from Problem.Benchmark.Problem import fitness as f
from util import util

def BorderCheck1(X,lb,ub,dim):
  for j in range(dim):
    if X[j]<lb[j]:
      X[j] = ub[j]
    elif X[j]>ub[j]:
      X[j] = lb[j]
  return X

def CaculateFitness1(X,function):
  fitness = f(function, X)
  return fitness

def CaculateFitness2(X,function):
  fitness = function(X)
  return fitness


def Intensity(pop,dim,GbestPositon,X):
  epsilon = 0.00000000000000022204
  di = np.zeros([pop,dim])
  S = np.zeros([pop,dim])
  I = np.zeros([pop,dim])
  for j in range(pop):
    
    if (j != pop-1):
      di[j]= [(e1 - e2)+epsilon for e1, e2 in zip(GbestPositon,X[j])]
      # di[j] = util.distEuclidiana(GbestPositon, X[j],False,False)+epsilon
      S[j]= [(e1 - e2)+epsilon for e1, e2 in zip(X[j],X[j+1])]
      # S[j] = util.distEuclidiana(X[j], X[j+1],False,False)+epsilon
      di[j] = np.power(di[j], 2)
      S[j]= np.power(S[j], 2)
    else:
      di[j]=[(e1 - e2)+epsilon for e1, e2 in zip(GbestPositon,X[j])]
      # di[j] = util.distEuclidiana(GbestPositon, X[j],False,False)+epsilon 
      S[j]=[(e1 - e2)+epsilon for e1, e2 in zip(X[j],X[1])]
      # S[j] = util.distEuclidiana(X[j], X[1],False,False)+epsilon
      di[j] = np.power(di[j], 2)
      S[j]= np.power(S[j], 2)    
  n = random.uniform(0.0 , 1.0)
  for i in range(pop):
    n = random.uniform(0.0 , 1.0)
    I[i] = n*S[i]/[(4*math.pi*di[i])]
  return I

def iterarHBA(Max_iter, t, dim, poblacion, GbestPositon, fitness, lb, ub, function, problem ): 
  C = 2                                          # constant in Eq. (3)
  beta = 6                            # the ability of HB to get the food  Eq.(4)
  Xnew = np.zeros([poblacion.__len__(),dim])
  alpha=C*math.exp(-t/Max_iter)            # density factor in Eq. (3)
  I=Intensity(poblacion.__len__(),dim,GbestPositon,poblacion);           # intensity in Eq. (2)
  for i in range(poblacion.__len__()):
    Vs=random.uniform(0.0 , 1.0)
    r3=random.uniform(0.0 , 1.0)
    r4=random.uniform(0.0 , 1.0)
    r5=random.uniform(0.0 , 1.0)
    r6=random.uniform(0.0 , 1.0)
    r7=random.uniform(0.0 , 1.0)
    if r6 <= 0.5:
      F = 1
    else:
      F = -1
    for j in range(dim):
      di=GbestPositon[j]-poblacion[i][j]
      if (Vs <0.5):                           # Digging phase Eq. (4)
        Xnew[i][j]=GbestPositon[j]+F*beta*I[i][j]*GbestPositon[j]+F*r3*alpha*(di)*np.abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5)))
      else:
        Xnew[i][j]=GbestPositon[j]+F*r7*alpha*di;    # Honey phase Eq. (6)
    
    
    if problem == 'ben':
      Xnew[i] = BorderCheck1(Xnew[i], lb, ub, dim)
      tempFitness = CaculateFitness1(Xnew[i], function)
    else:
      tempFitness = CaculateFitness2(Xnew[i], function)
      
    if (tempFitness <= fitness[i]):
      fitness[i] = tempFitness               
      poblacion[i] = Xnew[i] 
  return np.array(poblacion)
