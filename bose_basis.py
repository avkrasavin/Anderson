
# coding: utf-8

# In[ ]:
import numpy as np
from math import factorial as fact
def basis(m,n):
    '''Возвращает базис для Бозе-статистики. m - число узлов, n - число частиц'''
    R = fact(n + m - 1)/fact(n)/fact(m - 1)
    b = np.zeros((R,m))
    b[0,m-1] = n
    for i in range(R-1):
        j = m - 1
        while j > 0:
            if b[i,j] in range(2,n+1) :
                b[i+1,:] = b[i,:]
                b[i+1,j] = 0
                b[i+1,j-1] = b[i+1,j-1] + 1
                b[i+1,m-1] = b[i,j] - 1
                break
            elif b[i,j] > 0:
                b[i+1,:] = b[i,:]
                b[i+1,j-1] = b[i+1,j-1] + 1
                b[i+1,j] = b[i,j] - 1
                break
            j -= 1
    return b
def limit_basis(m,n,n_max):
    '''Возвращает базис для Бозе-статистики с ограничением числа частиц на узле. 
    m - число узлов, n - число частиц, n_max - максимальное число частиц на узле.'''
    # Размерность базиса
    R = fact(n + m - 1)/fact(n)/fact(m - 1)
    b = basis(m,n)
    f = np.zeros((R,m))
    j = 0
    # Откиддываем функции, в которых на узлах частиц больше n_max
    for i in range(b.shape[0]):
        if any(b[i] > n_max): 
            continue
        else:
            f[j] = b[i]
            j += 1
    return f[:j]

