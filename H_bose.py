
# coding: utf-8

# In[ ]:

from math import factorial as fact
import numpy as np
import copy
import bose_basis as bb
import progressbar
def up_down(number_to_up, number_to_down, func,n_max):
    '''Функция возвращает коэффициент и новую функцию.'''
    function = copy.copy(func)
    if function[number_to_down] == 0:
        return [0,function]
    elif function[number_to_up] == n_max:
        return [0,function]
    else:
        coef_down = np.sqrt(function[number_to_down])
        function[number_to_down] = function[number_to_down] - 1
        coef_up = np.sqrt(function[number_to_up] + 1)
        function[number_to_up] = function[number_to_up] + 1
        return [round(coef_down*coef_up,5), function]
def h_block(m,n,n_max,V=1):
    ''' Функция заполняет блоки матрицы Гамильтона.'''
    fi = bb.limit_basis(m,n,n_max)    
    # Переводим тип матрицы в целочисленную для корректной работы метода ''.join(...)
    fi = fi.astype(int)
    
    H = np.zeros((len(fi),len(fi)))
    # Создаем словарь для определения индекса функции 
    key = [''.join(map(str,fi[i])) for i in range(fi.shape[0])]
    value = range(len(key))
    indexes = dict(zip(key,value))
    
    
    for i in range(len(fi)):
        for j in range(1,m,1):
            # Перескоки с ванны на узел и обратно
            H[i,indexes[''.join(map(str,up_down(0,j,fi[i],n_max)[1]))]] += V*up_down(0,j,fi[i],n_max)[0]
            H[i,indexes[''.join(map(str,up_down(j,0,fi[i],n_max)[1]))]] += V*up_down(j,0,fi[i],n_max)[0]
    return H, len(fi)

def hamiltonian(m,n,n_max):
    '''Функция соединяет блоки в одну матрицу.'''
    R1 = 0
    H = np.zeros(((n_max+1)**m,(n_max+1)**m))
    bar = progressbar.ProgressBar()
    for n_current in bar(range(n)):
        h, R2 = h_block(m,n_current,n_max,1)
        H[R1:(R1+R2), R1:(R2+R1)] = h
        R1 += R2
    return H

