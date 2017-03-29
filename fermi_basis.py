
# coding: utf-8

# In[27]:

import numpy as np
from math import factorial as fact
import copy
import bose_basis as bb
def basis(m,n):
    '''Возвращает базис для Ферми-статистики без учета спина. m - число узлов, n - число частиц.'''
    # Размерность базиса
    R = fact(m)/fact(n)/fact(m-n)
    f = np.zeros((R,m))
    b = bb.basis(m,n)
    j = 0
    # Откиддываем функции, в которых на узлах частиц больше 1 
    for i in range(b.shape[0]):
        if any(b[i] > 1): 
            continue
        else:
            f[j] = b[i]
            j += 1
    return f
def basis_spin(m, n_up, n_down):
    '''Возвращает базис для Ферми-статистики с учетом спина. 
    m - число узлов, n_up - число спинов "вверх", n_dowm - число спинов "вниз".
    '''
    R = (fact(m)/fact(n_up)/fact(m-n_up))*(fact(m)/fact(n_down)/fact(m-n_down)) 
    fs = np.zeros((R,2*m))
    part_1 = basis(m,n_up)
    
    if n_up == n_down:
        part_2 = copy.copy(part_1)
    else:
        part_2 = basis(m,n_down)
    for i in range(basis(m,n_up).shape[0]):
        for j in range(basis(m,n_down).shape[0]):
            fs[i*basis(m,n_down).shape[0]+j] = np.concatenate((part_1[i],part_2[j]), axis=0)
    return fs
